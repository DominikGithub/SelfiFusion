package com.selfiefusion.segtest;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * M7 generative seam blend: LaMa inpainting (bundled in the APK) fills
 * the frame-crop seam bands of the saved fused still via ONNX Runtime.
 *
 * Model: OpenCV Zoo "inpainting_lama_2025jan.onnx" (fp16, 92.6MB,
 * Apache-2.0), interface verified with a local onnxruntime smoke test:
 *   input  "image"  1x3x512x512 float32, pixel/255 (range [0,1])
 *   input  "mask"   1x1x512x512 float32, 1 = hole, 0 = keep
 *   output "output" 1x3x512x512 float32, range [0,255], EXACTLY the input
 *          outside the hole (the blend is baked into the graph), so pasting
 *          back is seam-free.
 *
 * The model input is FIXED 512x512. Version history of the window
 * strategy, all decided from on-device results:
 *   v0.9.0 sliding full-res windows: visible seams between window
 *        generations (adjacent pastes don't blend) + minutes of latency.
 *   v0.9.2 one scaled window per band: seams and latency fixed, but a
 *        long band (~3000px) runs at ~1/5 resolution - device feedback:
 *        the "AI blend" reads as mere blurring, not generated content.
 *   v0.9.5 (this) TILED windows: each tile is sized by the band's CROSS
 *        axis only (typically near full resolution - 4-5x sharper than
 *        v0.9.2), and long bands are split along their long axis into
 *        several tiles. Consecutive tiles OVERLAP by
 *        TILE_OVERLAP_WINDOW_PX; the LATER tile pastes with an alpha ramp
 *        0->1 across the overlap onto the earlier tile's opaque paste,
 *        so neighbor generations crossfade cleanly (the v0.9.0 seam
 *        problem solved) instead of squeezing the whole band into one
 *        low-res window (the v0.9.2 blur problem). Short bands still
 *        run as a single full-res window.
 */
public final class Inpainter implements AutoCloseable {

    private static final String TAG = "SelfieFusion";
    private static final String MODEL_NAME = "inpaint.onnx";
    private static final long MIN_MODEL_BYTES = 10_000_000L;

    private static final int WINDOW = 512;          // model input size (fixed)
    // Hole long axis maps to at most WINDOW - 2*WINDOW_CONTEXT window
    // pixels: every side of the hole keeps >= WINDOW_CONTEXT of image
    // context inside the model input. 32, not 64: the context margin is
    // what shrinks the long-band window scale, and with 64 the saved
    // bands showed bubbly upscaled texture on device (the model had the
    // context, but at thumbnail resolution - the band hole itself
    // collapsed to a ~5-window-px slit). Extra band pixels beat extra
    // context.
    private static final int WINDOW_CONTEXT = 32;
    // v0.9.5 tiling: consecutive tiles of a long band overlap by this
    // many window px; the later tile's paste alpha ramps 0 -> 1 across
    // the overlap onto the earlier tile's opaque paste (crossfade of the
    // two generations - the fix for v0.9.0's inter-window seams).
    private static final int TILE_OVERLAP_WINDOW_PX = 64;
    private static final int FEATHER_RADIUS = 6;    // paste-back edge feather (window px)
    // Safety valve: one invalid band aborts the WHOLE blend (return 0) so
    // the caller's fallback ladder re-composites with the edge fade - a
    // partially blended still must never be saved. Real seam bands are far
    // below this cap; LaMa's design band is ~40% of the image.
    private static final float MAX_HOLE_AREA_FRACTION = 0.25f;
    private static final int N = WINDOW * WINDOW;
    private static final float INV_255 = 1f / 255f;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String imageInput;
    private final String maskInput;

    // Reused per-band state (allocation-free steady state).
    private final Bitmap windowBitmap = Bitmap.createBitmap(WINDOW, WINDOW, Bitmap.Config.ARGB_8888);
    private final Canvas windowCanvas = new Canvas(windowBitmap);
    private final Bitmap pasteBitmap = Bitmap.createBitmap(WINDOW, WINDOW, Bitmap.Config.ARGB_8888);
    private final int[] windowPixels = new int[N];
    private final int[] pastePixels = new int[N];
    private final float[] imageTensor = new float[3 * N];
    private final float[] maskTensor = new float[N];
    private final float[] blurTmp = new float[WINDOW];
    private final Rect srcRect = new Rect();
    private final RectF dstRectF = new RectF();
    private final Paint drawPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint pastePaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Map<String, OnnxTensor> inputs = new HashMap<>();

    /** Per-band progress callback (done/total seam bands). */
    public interface Progress {
        void accept(int done, int total);
    }

    private Inpainter(OrtEnvironment env, OrtSession session) {
        this.env = env;
        this.session = session;
        String image = "image";
        String mask = "mask";
        try {
            List<String> names = new ArrayList<>(session.getInputNames());
            for (String n : names) {
                if (n == null) {
                    continue;
                }
                String lower = n.toLowerCase(Locale.US);
                if (lower.contains("image") || lower.contains("img")) {
                    image = n;
                } else if (lower.contains("mask")) {
                    mask = n;
                }
            }
        } catch (Throwable t) {
            Log.w(TAG, "input name probe failed, using defaults: " + t);
        }
        this.imageInput = image;
        this.maskInput = mask;
    }

    /**
     * True if a usable model is available - already extracted to the
     * files dir or still bundled as an APK asset. Cheap (no extraction)
     * and safe on the UI thread; used by the menu toggle to warn early.
     */
    public static boolean hasModel(Context context) {
        if (extractedModel(context) != null) {
            return true;
        }
        try (InputStream in = context.getAssets().open(MODEL_NAME)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /** Previously extracted model copy in the files dir, or null. */
    private static File extractedModel(Context context) {
        File f = new File(context.getFilesDir(), MODEL_NAME);
        return f.isFile() && f.length() >= MIN_MODEL_BYTES ? f : null;
    }

    /**
     * Model file for the session: the bundled APK asset, extracted once
     * to the files dir (ORT then loads from a path, so the 92MB model
     * never sits on the Java heap). The tmp file + atomic rename keeps
     * a killed extraction from leaving a partial model behind. Null =
     * APK built without models/ or extraction failed; the caller falls
     * back to the edge fade.
     */
    private static File extractModel(Context context) {
        File cached = extractedModel(context);
        if (cached != null) {
            return cached;
        }
        File out = new File(context.getFilesDir(), MODEL_NAME);
        File tmp = new File(context.getFilesDir(), MODEL_NAME + ".tmp");
        try (InputStream in = context.getAssets().open(MODEL_NAME)) {
            Files.copy(in, tmp.toPath(), StandardCopyOption.REPLACE_EXISTING);
            if (tmp.length() < MIN_MODEL_BYTES) {
                throw new IOException("extracted model only " + tmp.length() + " bytes");
            }
            if (out.isFile() && !out.delete()) {
                throw new IOException("cannot replace stale " + out);
            }
            if (!tmp.renameTo(out)) {
                throw new IOException("rename to " + out + " failed");
            }
        } catch (IOException e) {
            tmp.delete();
            return null;
        }
        Log.i(TAG, "extracted inpainting model from APK asset: " + out.length() + " bytes");
        return out;
    }

    /** Loads the model; null = unavailable (caller falls back to the edge fade). */
    public static Inpainter create(Context context) {
        File model = extractModel(context);
        if (model == null) {
            Log.w(TAG, "no inpainting model (APK asset " + MODEL_NAME
                    + " missing or extraction failed) - seam bands keep the edge fade");
            return null;
        }
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession session = env.createSession(model.getAbsolutePath(),
                    new OrtSession.SessionOptions());
            Log.i(TAG, "inpainting model loaded: " + model + " (" + model.length() + " bytes)");
            return new Inpainter(env, session);
        } catch (Throwable t) {
            Log.w(TAG, "inpaint session create failed: " + t);
            return null;
        }
    }

    /**
     * Inpaints the given full-resolution hole rectangles in-place on
     * {@code fused}. v0.9.5: long bands are TILED into several
     * near-full-resolution windows (see the class javadoc); each window
     * is one inference. Returns the number of windows inpainted
     * (>= bands); 0 = nothing done, the caller must fall back to the
     * edge fade.
     */
    public int blend(Bitmap fused, List<Rect> holes, Progress progress) throws OrtException {
        final int w = fused.getWidth();
        final int h = fused.getHeight();
        if (w < 2 || h < 2 || holes == null || holes.isEmpty()) {
            return 0;
        }
        // Airtight validation: EVERY band must be processable. One bad
        // band aborts the whole blend (0 = caller re-composites with the
        // fade) instead of saving a partially blended still.
        ArrayList<Rect> bands = new ArrayList<>();
        for (Rect hole : holes) {
            Rect band = new Rect(Math.max(0, hole.left), Math.max(0, hole.top),
                    Math.min(w, hole.right), Math.min(h, hole.bottom));
            if (band.width() < 2 || band.height() < 2) {
                Log.w(TAG, "seam band rejected (empty): " + hole);
                return 0;
            }
            bands.add(band);
        }
        // v0.9.5 tile planning: split each band along its long axis into
        // windows sized by the CROSS axis (near full resolution), with
        // overlap for the paste crossfade. The oversized-band guard now
        // checks each TILE - the v0.9.4 outpaint growth makes single
        // bands exceed the old whole-band cap while every tile stays
        // window-sized.
        ArrayList<Tile> tiles = new ArrayList<>();
        for (Rect band : bands) {
            planTiles(tiles, band);
        }
        for (Tile t : tiles) {
            if (t.hole.width() * (long) t.hole.height()
                    > MAX_HOLE_AREA_FRACTION * w * h) {
                Log.w(TAG, "seam tile rejected (oversized): " + t.hole);
                return 0;
            }
        }
        Canvas fusedCanvas = new Canvas(fused);
        final int total = tiles.size();
        int done = 0;
        for (Tile t : tiles) {
            inpaintWindow(fusedCanvas, fused, t);
            done++;
            if (progress != null) {
                progress.accept(done, total);
            }
        }
        return done;
    }

    /** One planned model window of a band (see the class javadoc). */
    private static final class Tile {
        final Rect hole;          // full-res hole rectangle of this tile
        final boolean vertical;   // band long axis is vertical
        final boolean rampStart;  // paste alpha ramps 0->1 at the long-axis
                                 // start (crossfade with the previous tile)

        Tile(Rect hole, boolean vertical, boolean rampStart) {
            this.hole = hole;
            this.vertical = vertical;
            this.rampStart = rampStart;
        }
    }

    /**
     * Splits one band into tiles along its long axis. Tile scale is set
     * by the band's CROSS axis only (the hole must fit WINDOW - 2*
     * WINDOW_CONTEXT window px across, with context on both sides);
     * along the long axis each tile's hole spans up to the same budget
     * and consecutive tiles overlap by TILE_OVERLAP_WINDOW_PX/scale.
     * Bands that fit one window stay one tile (no ramp - single pass).
     */
    private static void planTiles(ArrayList<Tile> out, Rect band) {
        int cross = Math.min(band.width(), band.height());
        int longAx = Math.max(band.width(), band.height());
        boolean vertical = band.height() >= band.width();
        float scale = Math.min(1f, (WINDOW - 2 * WINDOW_CONTEXT) / (float) cross);
        float maxHoleLong = (WINDOW - 2 * WINDOW_CONTEXT) / scale;
        if (longAx <= maxHoleLong) {
            out.add(new Tile(new Rect(band), vertical, false));
            return;
        }
        float overlap = TILE_OVERLAP_WINDOW_PX / scale;
        int n = (int) Math.ceil((longAx - overlap) / (maxHoleLong - overlap));
        if (n < 2) {
            n = 2;
        }
        float step = (longAx - overlap) / n;
        for (int i = 0; i < n; i++) {
            int lo = Math.round(i * step);
            int hi = i == n - 1 ? longAx : Math.round(Math.min(longAx, lo + step + overlap));
            if (hi - lo < 8) {
                continue; // rounding guard at the band end
            }
            Rect r = vertical
                    ? new Rect(band.left, band.top + lo, band.right, band.top + hi)
                    : new Rect(band.left + lo, band.top, band.left + hi, band.bottom);
            out.add(new Tile(r, vertical, i > 0));
        }
    }

    /**
     * One model window for one tile: draws the tile hole + context
     * (scaled by the CROSS-axis budget) into the 512x512 input, runs
     * ONE inference, pastes back with a single filtered draw. The mask
     * and the paste-back share the exact same (x0, y0, scale) mapping,
     * so the generated content lands on the hole with sub-pixel
     * registration. A tile that continues a previous one pastes with a
     * 0->1 alpha ramp over the overlap (crossfade of the two
     * generations); the FIRST tile of a band pastes opaque up to the
     * blurred border feather, like the v0.9.2 single window did.
     */
    private void inpaintWindow(Canvas fusedCanvas, Bitmap fused, Tile tile) throws OrtException {
        final int w = fused.getWidth();
        final int h = fused.getHeight();
        final Rect band = tile.hole;
        // Window scale from the band's CROSS axis: the hole fits across
        // with context on both sides; along the long axis the window
        // simply shows as much context as the 512px input holds.
        float scale = Math.min(1f,
                (WINDOW - 2 * WINDOW_CONTEXT)
                        / (float) Math.min(band.width(), band.height()));
        float cover = WINDOW / scale;
        float x0 = Math.max(0f, Math.min(w - cover,
                (band.left + band.right) / 2f - cover / 2f));
        float y0 = Math.max(0f, Math.min(h - cover,
                (band.top + band.bottom) / 2f - cover / 2f));
        float x1 = Math.min(w, x0 + cover);
        float y1 = Math.min(h, y0 + cover);

        // Forward draw: the cover region scaled into the model window.
        // The <=0.5px rounding of the source rect only shifts the context
        // the model sees; the mask + paste below are unaffected.
        windowBitmap.eraseColor(0);
        srcRect.set(Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1));
        dstRectF.set(0, 0, (x1 - x0) * scale, (y1 - y0) * scale);
        windowCanvas.drawBitmap(fused, srcRect, dstRectF, drawPaint);
        windowBitmap.getPixels(windowPixels, 0, WINDOW, 0, 0, WINDOW, WINDOW);

        // Hole mask (1 = inpaint) in window coordinates, clipped to the
        // drawn region (the window padding must stay 0 = keep) and
        // enforced >= 1px so a thin band can never round away.
        Arrays.fill(maskTensor, 0f);
        int dstW = Math.min(WINDOW, Math.round((x1 - x0) * scale));
        int dstH = Math.min(WINDOW, Math.round((y1 - y0) * scale));
        int mx0 = Math.max(0, Math.min(dstW - 1, Math.round((band.left - x0) * scale)));
        int mx1 = Math.max(mx0 + 1, Math.min(dstW, Math.round((band.right - x0) * scale)));
        int my0 = Math.max(0, Math.min(dstH - 1, Math.round((band.top - y0) * scale)));
        int my1 = Math.max(my0 + 1, Math.min(dstH, Math.round((band.bottom - y0) * scale)));
        for (int y = my0; y < my1; y++) {
            int row = y * WINDOW;
            Arrays.fill(maskTensor, row + mx0, row + mx1, 1f);
        }
        // v0.9.5 crossfade: a tile that continues a previous one ramps
        // its paste alpha 0 -> 1 over the first TILE_OVERLAP_WINDOW_PX of
        // its hole along the long axis. The previous tile already pasted
        // OPAQUE generated content there (its own hole extends into this
        // overlap), so the ramp blends the two generations with no gap
        // and no visible seam - the fix for v0.9.0's sliding-window edges.
        if (tile.rampStart) {
            int o = TILE_OVERLAP_WINDOW_PX;
            if (tile.vertical) {
                for (int y = my0; y < my1; y++) {
                    float f = (y - my0) / (float) o;
                    if (f > 1f) {
                        break; // rest of the tile pastes opaque
                    }
                    if (f < 0f) {
                        f = 0f;
                    }
                    int row = y * WINDOW;
                    for (int x = mx0; x < mx1; x++) {
                        maskTensor[row + x] *= f;
                    }
                }
            } else {
                for (int x = mx0; x < mx1; x++) {
                    float f = (x - mx0) / (float) o;
                    if (f > 1f) {
                        break;
                    }
                    if (f < 0f) {
                        f = 0f;
                    }
                    for (int y = my0; y < my1; y++) {
                        maskTensor[y * WINDOW + x] *= f;
                    }
                }
            }
        }

        // Image tensor: NCHW, RGB, pixel/255.
        for (int i = 0; i < N; i++) {
            int p = windowPixels[i];
            imageTensor[i] = ((p >> 16) & 0xFF) * INV_255;
            imageTensor[N + i] = ((p >> 8) & 0xFF) * INV_255;
            imageTensor[2 * N + i] = (p & 0xFF) * INV_255;
        }

        try (OnnxTensor tImage = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(imageTensor), new long[]{1, 3, WINDOW, WINDOW});
             OnnxTensor tMask = OnnxTensor.createTensor(env,
                     FloatBuffer.wrap(maskTensor), new long[]{1, 1, WINDOW, WINDOW});
             OrtSession.Result result = session.run(buildInputs(tImage, tMask))) {
            OnnxTensor out = (OnnxTensor) result.get(0);
            FloatBuffer o = out.getFloatBuffer();
            // Paste-back alpha: blurred hole mask. Outside the hole the
            // model output is (at window scale) the input, so the feather
            // only softens the band border; after the upscale it doubles
            // as the soft transition between the generated band content
            // and the sharp full-res surroundings.
            boxBlurInPlace(maskTensor, FEATHER_RADIUS);
            for (int i = 0; i < N; i++) {
                int a = Math.round(maskTensor[i] * 255f);
                if (a <= 0) {
                    pastePixels[i] = 0;
                    continue;
                }
                int r = clamp255(o.get(i));
                int g = clamp255(o.get(N + i));
                int b = clamp255(o.get(2 * N + i));
                pastePixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
            }
            pasteBitmap.setPixels(pastePixels, 0, WINDOW, 0, 0, WINDOW, WINDOW);
            // Paste back: one filtered draw maps window (u, v) onto image
            // (x0 + u/scale, y0 + v/scale). The window padding is alpha 0
            // and draws nothing.
            dstRectF.set(x0, y0, x0 + cover, y0 + cover);
            fusedCanvas.drawBitmap(pasteBitmap, null, dstRectF, pastePaint);
        }
    }

    private Map<String, OnnxTensor> buildInputs(OnnxTensor image, OnnxTensor mask) {
        inputs.clear();
        inputs.put(imageInput, image);
        inputs.put(maskInput, mask);
        return inputs;
    }

    /** Separable sliding-window box blur on the WINDOW x WINDOW mask (in place). */
    private void boxBlurInPlace(float[] data, int radius) {
        if (radius < 1) {
            return;
        }
        float inv = 1f / (2 * radius + 1);
        for (int y = 0; y < WINDOW; y++) {
            int row = y * WINDOW;
            System.arraycopy(data, row, blurTmp, 0, WINDOW);
            float sum = blurTmp[0] * (radius + 1);
            for (int x = 1; x <= radius; x++) {
                sum += blurTmp[Math.min(x, WINDOW - 1)];
            }
            for (int x = 0; x < WINDOW; x++) {
                data[row + x] = sum * inv;
                sum += blurTmp[Math.min(x + radius + 1, WINDOW - 1)]
                        - blurTmp[Math.max(x - radius, 0)];
            }
        }
        for (int x = 0; x < WINDOW; x++) {
            for (int y = 0; y < WINDOW; y++) {
                blurTmp[y] = data[y * WINDOW + x];
            }
            float sum = blurTmp[0] * (radius + 1);
            for (int y = 1; y <= radius; y++) {
                sum += blurTmp[Math.min(y, WINDOW - 1)];
            }
            for (int y = 0; y < WINDOW; y++) {
                data[y * WINDOW + x] = sum * inv;
                sum += blurTmp[Math.min(y + radius + 1, WINDOW - 1)]
                        - blurTmp[Math.max(y - radius, 0)];
            }
        }
    }

    private static int clamp255(float v) {
        int i = Math.round(v);
        if (i < 0) {
            return 0;
        }
        return Math.min(i, 255);
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException ignored) {
        }
    }
}
