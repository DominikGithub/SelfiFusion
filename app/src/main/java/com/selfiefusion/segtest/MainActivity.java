package com.selfiefusion.segtest;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ConcurrentCamera;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.activity.ComponentActivity;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.segmentation.Segmentation;
import com.google.mlkit.vision.segmentation.SegmentationMask;
import com.google.mlkit.vision.segmentation.Segmenter;
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends ComponentActivity {

    private static final String TAG = "SelfieFusion";
    private static final int REQUEST_CAMERA = 1;
    private static final String[] MODE_NAMES = {"COMPOSITE", "RED BACKGROUND", "MASK HEATMAP", "OVERLAY OFF"};
    private static final float SEG_MIN = 0.30f;
    private static final float SEG_MAX = 0.70f;
    private static final int PERSON_MAX_LONG_SIDE = 640;
    // The segmentation model is by far the slowest part of the live pipeline
    // under dual-camera CPU contention. The person video is rendered from
    // EVERY camera frame using the cached mask; the segmenter only re-runs
    // when this much time has passed since the last completed run.
    private static final long SEG_INTERVAL_MS = 200;

    // --- Color match (M3): the cut-out person adopts the back scene's ---
    // --- lighting so the fused JPG stops looking like two photos from  ---
    // --- two different cameras.                                       ---
    // Statistics are collected on small downscaled copies (long side
    // capped at COLOR_STATS_SIDE); the transfer is applied as ONE native
    // ColorMatrix draw, never a full-res Java pixel loop. The named
    // constants are the tunables for device-test feedback:
    //   - luma: v0.8.0 mapped the person's mean luma EXACTLY onto the
    //     whole back photo's mean, which in practice DARKENED the person
    //     (front cameras expose faces brighter than the scene average),
    //     and the full std adoption crushed the midtones. Since v0.9.2
    //     the adoption is damped (MEAN_STRENGTH / CONTRAST_STRENGTH) and
    //     the person stays COLOR_SUBJECT_BRIGHTEN above the adopted
    //     anchor - the match follows the scene's tonal direction while
    //     the person keeps subject brightness (a face reads as the
    //     photo's subject, not its average).
    //   - chroma: the white-balance gap (mean U/V delta) applied damped
    //     (COLOR_CHROMA_STRENGTH) and capped (COLOR_CHROMA_SHIFT_MAX), so
    //     a dominantly colored scene (sunset, forest) cannot tint the
    //     face unnaturally.
    private static final int COLOR_STATS_SIDE = 192;
    private static final float COLOR_LUMA_SCALE_MIN = 0.60f;
    private static final float COLOR_LUMA_SCALE_MAX = 1.70f;
    // v0.9.2 luma rebalance tunables (see the luma bullet above).
    private static final float COLOR_LUMA_MEAN_STRENGTH = 0.50f;
    private static final float COLOR_LUMA_CONTRAST_STRENGTH = 0.60f;
    private static final float COLOR_SUBJECT_BRIGHTEN = 12f;
    private static final float COLOR_CHROMA_STRENGTH = 0.60f;
    private static final float COLOR_CHROMA_SHIFT_MAX = 16f;
    // Rec.709 luma; chroma axes of the transfer: U = R - Y, V = B - Y.
    private static final float LUMA_R = 0.2126f;
    private static final float LUMA_G = 0.7152f;
    private static final float LUMA_B = 0.0722f;

    // Frame-crop fade: where the person silhouette was CUT by the front
    // camera's frame (arms/torso/head reaching a frame border), the mask
    // is fully opaque up to that border and the composite would show a
    // hard rectangular seam ("pasted rectangle" look at the sides). The
    // cut-out alpha instead ramps smoothstep 0 -> 1 over this fraction of
    // the image width from the LEFT and RIGHT borders and of the height
    // from the TOP, so the person dissolves into the scene. The BOTTOM
    // border stays hard by design: the grounded-person rule glues it to
    // the frame bottom, where a real photo crops it. Named tunable in
    // the same spirit as SEG_MIN/SEG_MAX (device-test feedback).
    private static final float EDGE_FADE_FRACTION = 0.03f;
    // v0.9.3: the fade fraction is relative to the CUT-OUT, so a scaled-
    // down person got an imperceptibly thin ramp in the saved photo
    // (device feedback "softening isn't visible at all"). The saved
    // still instead uses the fraction that keeps the MAPPED ramp at
    // least this many px wide (capped so a tiny person never ghosts).
    private static final float EDGE_FADE_MIN_MAPPED_PX = 28f;
    private static final float EDGE_FADE_MAX_FRACTION = 0.15f;

    // --- M7 generative seam blend: where the person silhouette was CUT ---
    // --- by the front camera's frame and the cut-out's mapped edge lies  ---
    // --- INSIDE the back photo, the seam band (same geometry as the      ---
    // --- edge fade) is ADDITIONALLY re-synthesized by the bundled        ---
    // --- LaMa model from the person and scene patterns on both sides.    ---
    // --- Since v0.9.3 the fade ALWAYS runs (the AI band is painted over  ---
    // --- it), so every failure path - toggle off, no model file, session  ---
    // --- failure, detection miss, inference error - still leaves the     ---
    // --- faded (never hard) edge in the SAVED still.                     ---
    // Person-touches-frame detection threshold, scanned on the RAW mask
    // confidence (mask.conf) - the feather blur smears narrow cut strips
    // below the threshold, which silently skipped real seams in v0.9.x
    // (device feedback). A real cut reads ~0.95+, distant silhouette ~0.
    private static final float SEAM_PRESENCE_CONF = 0.85f;
    // Hole grows this many px beyond the band on all sides so the model
    // regenerates the existing edge pixels too (standard inpainting
    // practice - a hole flush with the content edge leaves a visible line).
    private static final int SEAM_DILATE = 16;

    // --- Outpaint-style hole growth (v0.9.4): a thin seam band only    ---
    // --- softens the cut line - device tests still read the long        ---
    // --- frame-cut seams as a "rectangular box". Where the mask's       ---
    // --- border-touch extent is LARGE (t-shirt cut at the left/right   ---
    // --- sides), the LaMa model should EXTEND the person content       ---
    // --- outward into the scene (outpainting semantics via the same    ---
    // --- inpaint model: the hole reaches into background pixels so     ---
    // --- the model paints person continuation there), not just         ---
    // --- repaint the seam strip. Growth is proportional to the        ---
    // --- border-touch extent (bigger intersection -> bigger hole),      ---
    // --- floored/capped and axis-limited so the single 512px model     ---
    // --- window (see Inpainter WINDOW/WINDOW_CONTEXT) keeps the       ---
    // --- cross-axis scale >= ~448/1600 (a typical grown band runs     ---
    // --- ~448/672 - below that the fill turns to bubbly upscaled      ---
    // --- texture). Top cuts stay at the thin band (hair               ---
    // --- edges already look good, user decision); bottom stays        ---
    // --- excluded (grounded-person rule).                              ---
    private static final float OUTPAINT_GROW_FRACTION = 0.15f;
    private static final int OUTPAINT_GROW_MAX_PX = 320;
    private static final int OUTPAINT_MIN_EXTEND_PX = 40;
    private static final int OUTPAINT_MAX_AXIS_PX = 1600;

    // Precomputed confidence -> output tables (256 entries): turns the
    // per-pixel float smoothstep/color math into a single array lookup.
    private static final int[] ALPHA_LUT = new int[256];
    private static final int[] RED_MASK_LUT = new int[256];
    private static final int[] HEAT_MASK_LUT = new int[256];

    static {
        for (int q = 0; q < 256; q++) {
            float t = (q / 255f - SEG_MIN) / (SEG_MAX - SEG_MIN);
            if (t < 0f) t = 0f;
            else if (t > 1f) t = 1f;
            float alpha = t * t * (3f - 2f * t);
            ALPHA_LUT[q] = Math.round(alpha * 255f);
            int c = ((255 - q) * 208) >> 8;
            RED_MASK_LUT[q] = (c << 24) | 0x00FF3030;
            HEAT_MASK_LUT[q] = (q << 24) | (q << 16) | (q << 8) | q;
        }
    }

    private static float smoothstep(float t) {
        if (t < 0f) t = 0f;
        else if (t > 1f) t = 1f;
        return t * t * (3f - 2f * t);
    }

    private PreviewView previewView;
    private PreviewView backPreviewView;
    private MaskOverlayView overlayView;
    private TextView infoView;
    private Button shutterButton;
    private ImageButton modeButton;
    private LinearLayout captureStatusCard;
    private ProgressBar captureSpinner;
    private TextView captureStatusText;
    private View flashView;
    private boolean captureUiActive = false;
    private final Runnable hideDoneCard = new Runnable() {
        @Override
        public void run() {
            captureStatusCard.animate().cancel();
            captureStatusCard.animate().alpha(0f).setDuration(250).start();
        }
    };

    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService composeExecutor = Executors.newSingleThreadExecutor();
    private Segmenter streamSegmenter;
    private Segmenter stillSegmenter;

    private ImageCapture frontImageCapture;
    private ImageCapture backImageCapture;
    private volatile boolean analyzing = false;
    private volatile boolean capturing = false;
    private volatile boolean frontCaptureBound = false;
    private volatile boolean backCaptureBound = false;
    private volatile boolean dualActive = false;
    private volatile boolean dualSupported = false;

    private byte[] pendingFrontJpeg;
    private int pendingFrontRotation;
    private byte[] pendingBackJpeg;
    private int pendingBackRotation;
    private MaskOverlayView.PersonPlacement livePlacement;
    private String dualStamp;

    private float[] smoothConf;
    private int smoothConfW;
    private int smoothConfH;
    private volatile long lastSegDone = 0;
    private volatile String lastMaskFormat = "-";

    private float[] conf;
    private int[] maskPixels;
    private Bitmap workFrame;
    private Canvas workCanvas;
    private final Matrix workMatrix = new Matrix();
    private final Paint workPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private int[] personPixels;
    private int[] maskXOfs;
    // Frame-crop fade ramps for the live person bitmap (same look as the
    // saved still's cut-out fade), rebuilt with the work bitmap.
    private float[] liveFadeX;
    private float[] liveFadeY;

    private boolean saveExtras = false;
    private boolean showStats = false;
    private boolean colorMatch = true;
    // M7 generative seam blend: default ON (user decision), only active
    // when the APK bundles the model and the session loads; the
    // v0.8.1 smoothstep fade is the automatic fallback.
    private boolean generativeBlend = true;
    private Inpainter inpainter;
    private boolean inpainterUnavailable = false;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private long frameCount = 0;
    private long fpsWindowStart = 0;
    private float lastFps = 0;
    private volatile int lastRotation = -1;
    private String cameraLabel = "?";

    private final Runnable fpsRunnable = new Runnable() {
        @Override
        public void run() {
            long now = SystemClock.elapsedRealtime();
            float fps = frameCount * 1000f / Math.max(1, now - fpsWindowStart);
            frameCount = 0;
            fpsWindowStart = now;
            lastFps = fps;
            if (showStats) {
                updateInfo();
            }
            mainHandler.postDelayed(this, 1000);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        backPreviewView = findViewById(R.id.backPreviewView);
        overlayView = findViewById(R.id.overlayView);
        infoView = findViewById(R.id.infoView);
        shutterButton = findViewById(R.id.shutterButton);
        modeButton = findViewById(R.id.modeButton);
        captureStatusCard = findViewById(R.id.captureStatusCard);
        captureSpinner = findViewById(R.id.captureSpinner);
        captureStatusText = findViewById(R.id.captureStatusText);
        flashView = findViewById(R.id.flashView);

        streamSegmenter = Segmentation.getClient(new SelfieSegmenterOptions.Builder()
                .setDetectorMode(SelfieSegmenterOptions.STREAM_MODE)
                .build());
        stillSegmenter = Segmentation.getClient(new SelfieSegmenterOptions.Builder()
                .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
                .build());

        modeButton.setOnClickListener(v -> showModeMenu());
        shutterButton.setOnClickListener(v -> takeStill());

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    private void startCamera() {
        ProcessCameraProvider.getInstance(this).addListener(() -> {
            try {
                ProcessCameraProvider provider = ProcessCameraProvider.getInstance(this).get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                Preview backPreview = new Preview.Builder().build();
                backPreview.setSurfaceProvider(backPreviewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setResolutionSelector(new ResolutionSelector.Builder()
                                .setResolutionStrategy(new ResolutionStrategy(new Size(960, 540),
                                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                                .build())
                        .build();
                analysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

                frontImageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build();
                backImageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build();

                provider.unbindAll();
                dualSupported = deviceSupportsConcurrent();
                dualActive = dualSupported
                        && tryBindDual(provider, analysis, frontImageCapture, backPreview, backImageCapture);
                if (!dualActive) {
                    provider.unbindAll();
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA,
                            preview, analysis, frontImageCapture);
                    frontCaptureBound = true;
                    backCaptureBound = false;
                }

                cameraLabel = !dualActive
                        ? (dualSupported ? "front (dual bind failed)" : "front (no concam HW)")
                        : (frontCaptureBound && backCaptureBound ? "DUAL f+b"
                        : (frontCaptureBound ? "DUAL f+b (no back still)" : "DUAL f+b (no still)"));
                previewView.setVisibility(dualActive ? View.GONE : View.VISIBLE);
                backPreviewView.setVisibility(dualActive ? View.VISIBLE : View.GONE);
                overlayView.setDualActive(dualActive);
                overlayView.setMode(dualActive
                        ? MaskOverlayView.MODE_COMPOSITE : MaskOverlayView.MODE_RED_BG);

                frameCount = 0;
                fpsWindowStart = SystemClock.elapsedRealtime();
                mainHandler.removeCallbacks(fpsRunnable);
                mainHandler.post(fpsRunnable);
                updateInfo();
                Log.i(TAG, "camera up: dualActive=" + dualActive
                        + " dualSupported=" + dualSupported
                        + " frontStill=" + frontCaptureBound + " backStill=" + backCaptureBound);
            } catch (Exception e) {
                Toast.makeText(this, "Camera error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean deviceSupportsConcurrent() {
        if (Build.VERSION.SDK_INT < 30) {
            return false;
        }
        try {
            CameraManager cameraManager =
                    (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            for (Set<String> combo : cameraManager.getConcurrentCameraIds()) {
                boolean front = false;
                boolean back = false;
                for (String id : combo) {
                    Integer facing = cameraManager.getCameraCharacteristics(id)
                            .get(CameraCharacteristics.LENS_FACING);
                    if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                        front = true;
                    }
                    if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                        back = true;
                    }
                }
                if (front && back) {
                    return true;
                }
            }
        } catch (Throwable t) {
            Log.w(TAG, "concurrent camera check failed: " + t);
        }
        return false;
    }

    private boolean tryBindDual(ProcessCameraProvider provider, ImageAnalysis analysis,
                               ImageCapture frontCapture, Preview backPreview, ImageCapture backCapture) {
        try {
            UseCaseGroup frontGroup = new UseCaseGroup.Builder()
                    .addUseCase(analysis)
                    .addUseCase(frontCapture)
                    .build();
            UseCaseGroup backGroup = new UseCaseGroup.Builder()
                    .addUseCase(backPreview)
                    .addUseCase(backCapture)
                    .build();
            provider.unbindAll();
            provider.bindToLifecycle(Arrays.asList(
                    new ConcurrentCamera.SingleCameraConfig(
                            CameraSelector.DEFAULT_FRONT_CAMERA, frontGroup, this),
                    new ConcurrentCamera.SingleCameraConfig(
                            CameraSelector.DEFAULT_BACK_CAMERA, backGroup, this)));
            frontCaptureBound = true;
            backCaptureBound = true;
            return true;
        } catch (Throwable t) {
            Log.w(TAG, "dual bind (front: analysis+capture / back: preview+capture) failed: " + t);
        }
        try {
            UseCaseGroup frontGroup = new UseCaseGroup.Builder()
                    .addUseCase(analysis)
                    .addUseCase(frontCapture)
                    .build();
            UseCaseGroup backGroup = new UseCaseGroup.Builder()
                    .addUseCase(backPreview)
                    .build();
            provider.unbindAll();
            provider.bindToLifecycle(Arrays.asList(
                    new ConcurrentCamera.SingleCameraConfig(
                            CameraSelector.DEFAULT_FRONT_CAMERA, frontGroup, this),
                    new ConcurrentCamera.SingleCameraConfig(
                            CameraSelector.DEFAULT_BACK_CAMERA, backGroup, this)));
            frontCaptureBound = true;
            backCaptureBound = false;
            Log.i(TAG, "dual bind succeeded without back still capture");
            return true;
        } catch (Throwable t) {
            Log.w(TAG, "dual bind (front: analysis+capture / back: preview) failed: " + t);
        }
        try {
            UseCaseGroup frontGroup = new UseCaseGroup.Builder()
                    .addUseCase(analysis)
                    .build();
            UseCaseGroup backGroup = new UseCaseGroup.Builder()
                    .addUseCase(backPreview)
                    .build();
            provider.unbindAll();
            provider.bindToLifecycle(Arrays.asList(
                    new ConcurrentCamera.SingleCameraConfig(
                            CameraSelector.DEFAULT_FRONT_CAMERA, frontGroup, this),
                    new ConcurrentCamera.SingleCameraConfig(
                            CameraSelector.DEFAULT_BACK_CAMERA, backGroup, this)));
            frontCaptureBound = false;
            backCaptureBound = false;
            Log.i(TAG, "dual bind succeeded without still capture");
            return true;
        } catch (Throwable t) {
            Log.w(TAG, "dual bind (front: analysis / back: preview) failed: " + t);
        }
        provider.unbindAll();
        return false;
    }

    private void analyzeFrame(ImageProxy imageProxy) {
        if (analyzing || imageProxy.getImage() == null || capturing) {
            imageProxy.close();
            return;
        }
        int rotation = imageProxy.getImageInfo().getRotationDegrees();
        lastRotation = rotation;
        int viewMode = overlayView.getMode();
        if (viewMode == MaskOverlayView.MODE_OFF) {
            // Segmentation/compositing skipped; with stats enabled the FPS
            // number then shows the raw camera delivery rate (debug aid).
            imageProxy.close();
            if (showStats) {
                mainHandler.post(() -> frameCount++);
            }
            return;
        }
        analyzing = true;
        boolean wantPerson = dualActive && viewMode == MaskOverlayView.MODE_COMPOSITE;
        boolean wantMaskPixels = viewMode == MaskOverlayView.MODE_RED_BG
                || viewMode == MaskOverlayView.MODE_HEATMAP;
        boolean redMode = viewMode == MaskOverlayView.MODE_RED_BG;

        // The cached mask is usable for a fast (segmentation-free) render if
        // it exists and its aspect still matches this frame's orientation.
        boolean maskUsable = smoothConf != null && smoothConfW > 0 && smoothConfH > 0;
        if (wantPerson && maskUsable) {
            int frameW = imageProxy.getWidth();
            int frameH = imageProxy.getHeight();
            int uprightW = (rotation == 90 || rotation == 270) ? frameH : frameW;
            int uprightH = (rotation == 90 || rotation == 270) ? frameW : frameH;
            if (Math.abs(uprightW / (float) uprightH
                    - smoothConfW / (float) smoothConfH) > 0.02f) {
                maskUsable = false;
            }
        }
        boolean needSeg = !maskUsable
                || SystemClock.elapsedRealtime() - lastSegDone >= SEG_INTERVAL_MS;
        if (!needSeg) {
            if (wantPerson) {
                renderFastFrame(imageProxy, rotation);
            } else {
                // RED/HEAT need no per-frame RGB; the mask refresh arrives
                // with the next segmentation run.
                analyzing = false;
                imageProxy.close();
            }
            return;
        }

        InputImage input = InputImage.fromMediaImage(imageProxy.getImage(), rotation);
        streamSegmenter.process(input)
                .addOnSuccessListener(composeExecutor, mask -> {
                    Bitmap person = null;
                    Bitmap frame = null;
                    boolean proxyClosed = false;
                    try {
                        // Copy the mask out of the ML Kit buffer (last use of it).
                        ByteBuffer buffer = mask.getBuffer();
                        int maskW = mask.getWidth();
                        int maskH = mask.getHeight();
                        int count = maskW * maskH;
                        boolean floatMask = buffer.remaining() >= count * 4L;
                        if (conf == null || conf.length != count) {
                            conf = new float[count];
                        }
                        if (floatMask) {
                            buffer.asFloatBuffer().get(conf, 0, count);
                        } else {
                            for (int i = 0; i < count; i++) {
                                conf[i] = (buffer.get() & 0xFF) / 255f;
                            }
                            buffer.rewind();
                        }
                        // Last use of the camera frame: native YUV->RGB copy.
                        if (wantPerson) {
                            frame = imageProxy.toBitmap();
                        }
                        // Release the camera pipeline immediately: delivery +
                        // ML Kit inference of the next frame now overlap the
                        // heavy work below instead of waiting for it.
                        imageProxy.close();
                        proxyClosed = true;
                        analyzing = false;
                        lastSegDone = SystemClock.elapsedRealtime();

                        if (smoothConf == null || smoothConfW != maskW || smoothConfH != maskH) {
                            smoothConf = new float[count];
                            System.arraycopy(conf, 0, smoothConf, 0, count);
                            smoothConfW = maskW;
                            smoothConfH = maskH;
                        } else {
                            for (int i = 0; i < count; i++) {
                                smoothConf[i] = smoothConf[i] * 0.55f + conf[i] * 0.45f;
                            }
                        }
                        boxBlur(smoothConf, maskW, maskH, 2);

                        int[] colored = null;
                        if (wantMaskPixels) {
                            colored = colorizeMask(smoothConf, maskW, maskH, redMode);
                        }
                        if (wantPerson) {
                            try {
                                person = composePerson(frame, smoothConf, maskW, maskH, rotation);
                            } catch (Throwable t) {
                                Log.w(TAG, "person compositing failed: " + t);
                            }
                        }
                        String format = floatMask ? "f32" : "u8";
                        lastMaskFormat = format;
                        final int[] coloredOut = colored;
                        final Bitmap personOut = person;
                        mainHandler.post(() -> {
                            overlayView.updateFrame(coloredOut, maskW, maskH, format, personOut);
                            frameCount++;
                        });
                    } catch (Throwable t) {
                        Log.w(TAG, "frame post-processing failed: " + t);
                    } finally {
                        if (!proxyClosed) {
                            imageProxy.close();
                        }
                        analyzing = false;
                        if (frame != null) {
                            frame.recycle();
                        }
                    }
                })
                .addOnFailureListener(composeExecutor, e -> {
                    Log.w(TAG, "stream segmentation failed: " + e);
                    imageProxy.close();
                    analyzing = false;
                });
    }

    /**
     * Fast per-frame render without segmentation: refreshes only the person
     * RGB from the current camera frame and re-applies the cached mask.
     * This is what makes the live composite smooth - its frame cost is just
     * the native YUV->RGB copy plus the small compose, instead of the
     * (much slower) ML Kit inference that only refreshes the mask.
     */
    private void renderFastFrame(ImageProxy imageProxy, int rotation) {
        final int maskW = smoothConfW;
        final int maskH = smoothConfH;
        composeExecutor.execute(() -> {
            Bitmap person = null;
            Bitmap frame = null;
            boolean proxyClosed = false;
            try {
                frame = imageProxy.toBitmap();
                imageProxy.close();
                proxyClosed = true;
                analyzing = false;
                try {
                    person = composePerson(frame, smoothConf, maskW, maskH, rotation);
                } catch (Throwable t) {
                    Log.w(TAG, "person compositing failed: " + t);
                }
                final Bitmap personOut = person;
                final String format = lastMaskFormat;
                mainHandler.post(() -> {
                    overlayView.updateFrame(null, maskW, maskH, format, personOut);
                    frameCount++;
                });
            } catch (Throwable t) {
                Log.w(TAG, "fast frame failed: " + t);
            } finally {
                if (!proxyClosed) {
                    imageProxy.close();
                }
                analyzing = false;
                if (frame != null) {
                    frame.recycle();
                }
            }
        });
    }

    private int[] colorizeMask(float[] data, int w, int h, boolean redMode) {
        int count = w * h;
        if (maskPixels == null || maskPixels.length != count) {
            maskPixels = new int[count];
        }
        int[] lut = redMode ? RED_MASK_LUT : HEAT_MASK_LUT;
        for (int i = 0; i < count; i++) {
            int q = (int) (data[i] * 255f + 0.5f);
            if (q < 0) q = 0;
            else if (q > 255) q = 255;
            maskPixels[i] = lut[q];
        }
        return maskPixels;
    }

    private Bitmap composePerson(Bitmap frame, float[] conf, int maskW, int maskH,
                                 int rotation) {
        if (frame == null) {
            return null;
        }
        // Work at (at most) PERSON_MAX_LONG_SIDE: the mask derives from a
        // 256x256 model, so a larger person bitmap adds no edge detail -
        // it only costs time and memory.
        float cap = PERSON_MAX_LONG_SIDE / (float) Math.max(maskW, maskH);
        if (cap > 1f) {
            cap = 1f;
        }
        int tw = Math.max(2, Math.round(maskW * cap));
        int th = Math.max(2, Math.round(maskH * cap));
        if (workFrame == null || workFrame.getWidth() != tw || workFrame.getHeight() != th) {
            if (workFrame != null) {
                workFrame.recycle();
            }
            workFrame = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888);
            workCanvas = new Canvas(workFrame);
            personPixels = new int[tw * th];
            maskXOfs = new int[tw];
            for (int x = 0; x < tw; x++) {
                maskXOfs[x] = Math.min(maskW - 1, x * maskW / tw);
            }
            // Frame-crop fade ramps (see EDGE_FADE_FRACTION): left/right
            // and top, so the live preview matches the saved still.
            int fadeW = Math.max(1, Math.round(tw * EDGE_FADE_FRACTION));
            int fadeH = Math.max(1, Math.round(th * EDGE_FADE_FRACTION));
            liveFadeX = new float[tw];
            for (int x = 0; x < tw; x++) {
                float fromLeft = x < fadeW ? smoothstep(x / (float) fadeW) : 1f;
                float fromRight = tw - 1 - x < fadeW
                        ? smoothstep((tw - 1 - x) / (float) fadeW) : 1f;
                liveFadeX[x] = Math.min(fromLeft, fromRight);
            }
            liveFadeY = new float[th];
            for (int y = 0; y < th; y++) {
                // Top only: the bottom edge stays hard (grounded person).
                liveFadeY[y] = y < fadeH ? smoothstep(y / (float) fadeH) : 1f;
            }
        }
        // Single filtered draw: rotate + scale into the reused work bitmap
        // (replaces the previous rotate bitmap -> scaled bitmap -> getPixels
        // chain of full-resolution passes and per-frame allocations).
        int rotatedW = (rotation == 90 || rotation == 270) ? frame.getHeight() : frame.getWidth();
        int rotatedH = (rotation == 90 || rotation == 270) ? frame.getWidth() : frame.getHeight();
        workMatrix.reset();
        workMatrix.postRotate(rotation);
        workMatrix.postScale(tw / (float) rotatedW, th / (float) rotatedH);
        if (rotation == 90) {
            workMatrix.postTranslate(tw, 0);
        } else if (rotation == 180) {
            workMatrix.postTranslate(tw, th);
        } else if (rotation == 270) {
            workMatrix.postTranslate(0, th);
        }
        workCanvas.drawBitmap(frame, workMatrix, workPaint);
        workFrame.getPixels(personPixels, 0, tw, 0, 0, tw, th);

        for (int y = 0; y < th; y++) {
            int maskRow = Math.min(maskH - 1, y * maskH / th) * maskW;
            int row = y * tw;
            float fadeY = liveFadeY[y];
            for (int x = 0; x < tw; x++) {
                int q = (int) (conf[maskRow + maskXOfs[x]] * 255f + 0.5f);
                if (q < 0) q = 0;
                else if (q > 255) q = 255;
                int i = row + x;
                int alpha = (int) (ALPHA_LUT[q] * fadeY * liveFadeX[x] + 0.5f);
                personPixels[i] = (alpha << 24) | (personPixels[i] & 0x00FFFFFF);
            }
        }
        return Bitmap.createBitmap(personPixels, tw, th, Bitmap.Config.ARGB_8888);
    }

    private void showModeMenu() {
        PopupMenu popup = new PopupMenu(this, modeButton);
        popup.getMenuInflater().inflate(R.menu.overlay_menu, popup.getMenu());
        Menu menu = popup.getMenu();
        menu.findItem(R.id.mode_composite).setVisible(dualActive);
        int current = overlayView.getMode();
        int checkedId = current == MaskOverlayView.MODE_COMPOSITE ? R.id.mode_composite
                : current == MaskOverlayView.MODE_RED_BG ? R.id.mode_red
                : current == MaskOverlayView.MODE_HEATMAP ? R.id.mode_heat
                : R.id.mode_off;
        menu.findItem(checkedId).setChecked(true);
        menu.findItem(R.id.save_extras).setChecked(saveExtras);
        menu.findItem(R.id.color_match).setChecked(colorMatch);
        menu.findItem(R.id.generative_blend).setChecked(generativeBlend);
        menu.findItem(R.id.show_stats).setChecked(showStats);
        popup.setOnMenuItemClickListener(item -> {
            int id = item.getItemId();
            if (id == R.id.save_extras) {
                saveExtras = !saveExtras;
                item.setChecked(saveExtras);
                updateInfo();
                return true;
            }
            if (id == R.id.color_match) {
                colorMatch = !colorMatch;
                item.setChecked(colorMatch);
                updateInfo();
                return true;
            }
            if (id == R.id.generative_blend) {
                generativeBlend = !generativeBlend;
                item.setChecked(generativeBlend);
                if (generativeBlend && !Inpainter.hasModel(this)) {
                    Toast.makeText(this,
                            "No inpainting model in this APK - seams will use the edge fade",
                            Toast.LENGTH_LONG).show();
                }
                updateInfo();
                return true;
            }
            if (id == R.id.show_stats) {
                showStats = !showStats;
                item.setChecked(showStats);
                infoView.setVisibility(showStats ? View.VISIBLE : View.GONE);
                updateInfo();
                return true;
            }
            int newMode = id == R.id.mode_composite ? MaskOverlayView.MODE_COMPOSITE
                    : id == R.id.mode_red ? MaskOverlayView.MODE_RED_BG
                    : id == R.id.mode_heat ? MaskOverlayView.MODE_HEATMAP
                    : MaskOverlayView.MODE_OFF;
            overlayView.setMode(newMode);
            updateInfo();
            return true;
        });
        popup.show();
    }

    private void updateInfo() {
        if (!showStats) {
            return;
        }
        String text = String.format(Locale.US,
                "%s | %.0f FPS | mask %dx%d %s | rot %d | %s cam",
                MODE_NAMES[overlayView.getMode()], lastFps,
                overlayView.getMaskWidth(), overlayView.getMaskHeight(),
                overlayView.getMaskFormat(), lastRotation, cameraLabel);
        infoView.setText(text);
    }

    private void flash() {
        flashView.animate().cancel();
        flashView.setVisibility(View.VISIBLE);
        flashView.setAlpha(0.55f);
        flashView.animate().alpha(0f).setDuration(220)
                .withEndAction(() -> flashView.setVisibility(View.GONE))
                .start();
    }

    private void setShutterBusy(boolean busy) {
        shutterButton.setEnabled(!busy);
        shutterButton.animate().cancel();
        shutterButton.animate().alpha(busy ? 0.35f : 1f).setDuration(150).start();
    }

    private void showCaptureStage(String stage) {
        mainHandler.post(() -> {
            captureUiActive = true;
            mainHandler.removeCallbacks(hideDoneCard);
            captureStatusCard.animate().cancel();
            captureSpinner.setVisibility(View.VISIBLE);
            captureStatusText.setText(stage);
            captureStatusCard.setAlpha(1f);
            captureStatusCard.setVisibility(View.VISIBLE);
        });
    }

    private void finishCaptureUi(boolean ok, String message) {
        mainHandler.post(() -> {
            if (!captureUiActive) {
                return;
            }
            captureUiActive = false;
            setShutterBusy(false);
            if (ok) {
                captureSpinner.setVisibility(View.GONE);
                captureStatusText.setText("\u2713 " + message);
                mainHandler.postDelayed(hideDoneCard, 1600);
            } else {
                captureStatusCard.animate().cancel();
                captureStatusCard.setAlpha(1f);
                captureStatusCard.setVisibility(View.GONE);
            }
        });
    }

    private void takeStill() {
        if (frontImageCapture == null || capturing) {
            return;
        }
        if (dualActive && frontCaptureBound && backCaptureBound) {
            takeDualStill();
        } else if (frontCaptureBound) {
            takeFrontStill();
        } else {
            Toast.makeText(this, "Still capture unavailable in this camera mode",
                    Toast.LENGTH_SHORT).show();
        }
    }

    private void takeFrontStill() {
        capturing = true;
        flash();
        setShutterBusy(true);
        showCaptureStage("Capturing full-res photo...");
        frontImageCapture.takePicture(cameraExecutor, new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                try {
                    ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                    byte[] jpeg = new byte[buffer.remaining()];
                    buffer.get(jpeg);
                    int rotation = image.getImageInfo().getRotationDegrees();
                    image.close();

                    Bitmap photo = upright(jpeg, rotation, true);
                    if (photo == null) {
                        capturing = false;
                        finishCaptureUi(false, null);
                        runOnUiThread(() -> Toast.makeText(MainActivity.this,
                                "Failed to decode captured image", Toast.LENGTH_LONG).show());
                        return;
                    }
                    showCaptureStage("Segmenting person...");
                    stillSegmenter.process(InputImage.fromBitmap(photo, 0))
                            .addOnSuccessListener(cameraExecutor, mask -> saveResult(jpeg, photo, mask))
                            .addOnFailureListener(cameraExecutor, e -> {
                                photo.recycle();
                                finishCaptureUi(false, null);
                                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                                        "Segmentation failed: " + e.getMessage(), Toast.LENGTH_LONG).show());
                            })
                            .addOnCompleteListener(cameraExecutor, r -> capturing = false);
                } catch (Throwable t) {
                    capturing = false;
                    finishCaptureUi(false, null);
                    runOnUiThread(() -> Toast.makeText(MainActivity.this,
                            "Capture error: " + t.getMessage(), Toast.LENGTH_LONG).show());
                }
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                capturing = false;
                finishCaptureUi(false, null);
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Capture failed: " + exception.getMessage(), Toast.LENGTH_LONG).show());
            }
        });
    }

    private void takeDualStill() {
        capturing = true;
        pendingFrontJpeg = null;
        pendingBackJpeg = null;
        // Freeze the on-screen person placement at press time (UI thread)
        // so the saved still reproduces exactly what the user arranged.
        livePlacement = overlayView.snapshotPlacement();
        dualStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        flash();
        setShutterBusy(true);
        showCaptureStage("Capturing both cameras...");
        frontImageCapture.takePicture(cameraExecutor, new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                pendingFrontJpeg = new byte[buffer.remaining()];
                buffer.get(pendingFrontJpeg);
                pendingFrontRotation = image.getImageInfo().getRotationDegrees();
                image.close();
                tryFinishDualStill();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Front capture failed: " + exception.getMessage(), Toast.LENGTH_LONG).show());
                capturing = false;
                finishCaptureUi(false, null);
            }
        });
        backImageCapture.takePicture(cameraExecutor, new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                pendingBackJpeg = new byte[buffer.remaining()];
                buffer.get(pendingBackJpeg);
                pendingBackRotation = image.getImageInfo().getRotationDegrees();
                image.close();
                tryFinishDualStill();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Back capture failed: " + exception.getMessage(), Toast.LENGTH_LONG).show());
                capturing = false;
                finishCaptureUi(false, null);
            }
        });
    }

    private void tryFinishDualStill() {
        if (pendingFrontJpeg == null || pendingBackJpeg == null) {
            return;
        }
        byte[] frontJpeg = pendingFrontJpeg;
        byte[] backJpeg = pendingBackJpeg;
        int frontRotation = pendingFrontRotation;
        int backRotation = pendingBackRotation;
        pendingFrontJpeg = null;
        pendingBackJpeg = null;
        if (!capturing) {
            return;
        }
        try {
            Bitmap front = upright(frontJpeg, frontRotation, true);
            if (front == null) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Failed to decode front image", Toast.LENGTH_LONG).show());
                capturing = false;
                finishCaptureUi(false, null);
                return;
            }
            showCaptureStage("Segmenting person...");
            stillSegmenter.process(InputImage.fromBitmap(front, 0))
                    .addOnSuccessListener(cameraExecutor, mask ->
                            processFused(front, frontJpeg, backJpeg, backRotation, mask,
                                    livePlacement))
                    .addOnFailureListener(cameraExecutor, e -> {
                        front.recycle();
                        runOnUiThread(() -> Toast.makeText(MainActivity.this,
                                "Segmentation failed: " + e.getMessage(), Toast.LENGTH_LONG).show());
                        capturing = false;
                        finishCaptureUi(false, null);
                    });
        } catch (Throwable t) {
            capturing = false;
            finishCaptureUi(false, null);
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Fusion error: " + t.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

private void processFused(Bitmap front, byte[] frontJpeg, byte[] backJpeg,
                              int backRotation, SegmentationMask mask,
                              MaskOverlayView.PersonPlacement placement) {
        Cutout cutout = null;
        Bitmap person = null;
        Bitmap back = null;
        Bitmap fused = null;
        showCaptureStage("Fusing + saving...");
        try {
            back = upright(backJpeg, backRotation, false);
            if (back == null) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Failed to decode back image", Toast.LENGTH_LONG).show());
                finishCaptureUi(false, null);
                return;
            }
            // M7 (v0.9.3): the cut-out ALWAYS gets the frame-crop fade - in
            // AI mode too. The AI band regeneration overwrites the fade
            // inside the seam band, so it costs nothing when the AI runs,
            // but it guarantees the SAVED still never shows a hard
            // frame-crop seam when the AI blend is off, the model is
            // missing/unloadable, presence detection misses the cut, or
            // the blend fails: every AI failure just leaves the already-
            // faded composite (the v0.9.1 re-composite fallback is gone).
            // The fade width keeps a minimum in PLACED pixels so a scaled-
            // down person still gets a perceptible ramp.
            final float fadeFraction = edgeFadeFraction(front, back, placement);
            MaskData maskData = readMask(mask);
            cutout = buildCutout(front, maskData, fadeFraction);
            if (cutout == null) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Segmentation mask invalid", Toast.LENGTH_LONG).show());
                finishCaptureUi(false, null);
                return;
            }
            person = cutout.bitmap;
            final int cutW = person.getWidth();
            final int cutH = person.getHeight();
            Matrix matrix = placementForCutout(cutW, cutH, back, placement);
            // M3 color match: adopt the back scene's lighting before the
            // composite (dims unchanged -> placement math unaffected).
            if (colorMatch) {
                person = colorMatchToScene(person, back);
            }
            fused = compositeFused(back, person, matrix);

            // M7: regenerate the frame-crop seam bands on top of the fade
            // (one scaled 512x512 model window per band). All failure
            // paths only log - the fade underneath is the guaranteed edge
            // treatment of the saved still.
            if (generativeBlend && !inpainterUnavailable) {
                showCaptureStage("Preparing AI blend...");
                Inpainter inp = ensureInpainter();
                if (inp != null) {
                    ArrayList<Rect> holes = buildSeamHoleRects(fused.getWidth(), fused.getHeight(),
                            matrix, cutW, cutH, cutout, fadeFraction);
                    if (!holes.isEmpty()) {
                        try {
                            int windows = inp.blend(fused, holes,
                                    (done, total) -> showCaptureStage(
                                            "AI edge blend " + done + "/" + total + "..."));
                            Log.i(TAG, "generative seam blend: " + windows + " window(s) for "
                                    + holes.size() + " seam band(s) inpainted");
                        } catch (Throwable t) {
                            Log.w(TAG, "generative seam blend failed (edge fade stays): " + t);
                        }
                    } else {
                        Log.i(TAG, "no seam bands: person not cut by the front frame in-photo");
                    }
                }
            }

            if (saveExtras) {
                saveJpeg(frontJpeg, "SF_front_" + dualStamp + ".jpg");
                saveJpeg(backJpeg, "SF_back_" + dualStamp + ".jpg");
                saveBitmap(person, "SF_cutout_" + dualStamp + ".png", "image/png", true);
            }
            Uri fusedUri = saveBitmap(fused, "SF_fused_" + dualStamp + ".jpg", "image/jpeg", false);
            openSavedImagePreview(fusedUri, "image/jpeg");
            finishCaptureUi(true, saveExtras
                    ? "Fused selfie + materials saved to Pictures/SelfieFusion"
                    : "Fused selfie saved to Pictures/SelfieFusion");
        } catch (OutOfMemoryError e) {
            finishCaptureUi(false, null);
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Out of memory processing fused still", Toast.LENGTH_LONG).show());
        } catch (Throwable t) {
            finishCaptureUi(false, null);
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Fusion error: " + t.getMessage(), Toast.LENGTH_LONG).show());
        } finally {
            if (front != null && !front.isRecycled()) {
                front.recycle();
            }
            if (person != null && !person.isRecycled()) {
                person.recycle();
            }
            if (back != null) {
                back.recycle();
            }
            if (fused != null) {
                fused.recycle();
            }
            capturing = false;
        }
    }

    /** Fresh fused composite: back photo + person cut-out through the placement matrix. */
    private static Bitmap compositeFused(Bitmap back, Bitmap person, Matrix matrix) {
        Bitmap fused = Bitmap.createBitmap(back.getWidth(), back.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(fused);
        canvas.drawBitmap(back, 0, 0, null);
        canvas.drawBitmap(person, matrix, new Paint(Paint.FILTER_BITMAP_FLAG));
        return fused;
    }

    /**
     * Frame-crop fade width for the saved still: EDGE_FADE_FRACTION of the
     * cut-out, but at least EDGE_FADE_MIN_MAPPED_PX in the PLACED photo -
     * a scaled-down person otherwise gets an imperceptibly thin ramp (the
     * v0.9.3 device feedback "softening isn't visible at all"). Mirrors
     * the scale math of placementForCutout (placement-driven, height
     * matched); capped at EDGE_FADE_MAX_FRACTION so a very tiny person
     * never ghosts away. The live preview keeps the plain fraction.
     */
    private static float edgeFadeFraction(Bitmap front, Bitmap back,
                                          MaskOverlayView.PersonPlacement placement) {
        float cutW = front.getWidth();
        float cutH = front.getHeight();
        float scale;
        if (placement == null || !placement.valid) {
            scale = Math.max(back.getWidth() / cutW, back.getHeight() / cutH);
        } else {
            float fit = Math.max(placement.viewWidth / (float) placement.bitmapWidth,
                    placement.viewHeight / (float) placement.bitmapHeight);
            float personHView = placement.bitmapHeight * fit * placement.scale;
            float viewToBack = 1f / Math.max(placement.viewWidth / (float) back.getWidth(),
                    placement.viewHeight / (float) back.getHeight());
            scale = personHView * viewToBack / cutH;
        }
        float fraction = EDGE_FADE_MIN_MAPPED_PX / (cutW * scale);
        if (fraction < EDGE_FADE_FRACTION) {
            return EDGE_FADE_FRACTION;
        }
        return Math.min(fraction, EDGE_FADE_MAX_FRACTION);
    }

    /**
     * Transform that reproduces the on-screen person placement inside the
     * saved back photo. The back preview (PreviewView FILL_CENTER) shows a
     * centered crop of the back photo, so view coordinates map linearly
     * into photo coordinates; the person height arranged on screen is kept
     * and the bottom edge is glued to the bottom of the photo (grounded
     * person, never floating). Takes cut-out DIMS, not the bitmap: the
     * bitmap may already have been recycled by the color match when the
     * placement matrix is built.
     */
    private static Matrix placementForCutout(int cutW, int cutH, Bitmap back,
                                             MaskOverlayView.PersonPlacement placement) {
        int backW = back.getWidth();
        int backH = back.getHeight();
        Matrix matrix = new Matrix();
        if (placement == null || !placement.valid) {
            // Fallback: cover the back photo, bottom edge glued to the frame.
            float scale = Math.max(backW / (float) cutW, backH / (float) cutH);
            matrix.postScale(scale, scale);
            matrix.postTranslate((backW - cutW * scale) / 2f, backH - cutH * scale);
            return matrix;
        }
        float fit = Math.max(placement.viewWidth / (float) placement.bitmapWidth,
                placement.viewHeight / (float) placement.bitmapHeight);
        float personHView = placement.bitmapHeight * fit * placement.scale;
        float viewToBack = 1f / Math.max(placement.viewWidth / (float) backW,
                placement.viewHeight / (float) backH);
        float scale = personHView * viewToBack / cutH;
        float centerXPhoto = (placement.centerX - placement.viewWidth / 2f) * viewToBack
                + backW / 2f;
        float left = centerXPhoto - cutW * scale / 2f;
        float top = backH - cutH * scale;
        matrix.postScale(scale, scale);
        matrix.postTranslate(left, top);
        return matrix;
    }

    private static void boxBlur(float[] data, int w, int h, int radius) {
        if (data == null || w < 2 || h < 2 || radius < 1) {
            return;
        }
        float inv = 1f / (2 * radius + 1);
        float[] tmp = new float[Math.max(w, h)];
        for (int y = 0; y < h; y++) {
            int row = y * w;
            System.arraycopy(data, row, tmp, 0, w);
            float sum = tmp[0] * (radius + 1);
            for (int x = 1; x <= radius; x++) {
                sum += tmp[Math.min(x, w - 1)];
            }
            for (int x = 0; x < w; x++) {
                data[row + x] = sum * inv;
                int add = Math.min(x + radius + 1, w - 1);
                int rem = Math.max(x - radius, 0);
                sum += tmp[add] - tmp[rem];
            }
        }
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                tmp[y] = data[y * w + x];
            }
            float sum = tmp[0] * (radius + 1);
            for (int y = 1; y <= radius; y++) {
                sum += tmp[Math.min(y, h - 1)];
            }
            for (int y = 0; y < h; y++) {
                data[y * w + x] = sum * inv;
                int add = Math.min(y + radius + 1, h - 1);
                int rem = Math.max(y - radius, 0);
                sum += tmp[add] - tmp[rem];
            }
        }
    }

    /** Raw ML Kit mask, cached once per capture (see readMask). */
    private static class MaskData {
        final float[] conf;
        final int width;
        final int height;

        MaskData(float[] conf, int width, int height) {
            this.conf = conf;
            this.width = width;
            this.height = height;
        }
    }

    /**
     * Reads the segmentation mask ONCE into a cache: the ML Kit buffer is
     * not guaranteed to stay readable later - buildCutout consumes the
     * cache (and its presence scan reads the RAW conf directly), so the
     * buffer is never touched again after this.
     */
    private static MaskData readMask(SegmentationMask mask) {
        ByteBuffer buffer = mask.getBuffer();
        int width = mask.getWidth();
        int height = mask.getHeight();
        int count = width * height;
        float[] conf = new float[Math.max(count, 1)];
        if (count > 0 && buffer.remaining() >= count * 4L) {
            buffer.asFloatBuffer().get(conf, 0, count);
        } else {
            for (int i = 0; i < count; i++) {
                conf[i] = (buffer.get() & 0xFF) / 255f;
            }
        }
        return new MaskData(conf, width, height);
    }

    /** Cut-out plus the M7 frame-crop geometry (see Cutout fields). */
    private static class Cutout {
        final Bitmap bitmap;
        final int maskWidth;
        final int maskHeight;
        // Person-touches-frame ranges in MASK coords (inclusive min..max);
        // -1/-1 = the silhouette does not touch that frame border.
        final int leftY0, leftY1, rightY0, rightY1, topX0, topX1;

        Cutout(Bitmap bitmap, int maskWidth, int maskHeight,
               int leftY0, int leftY1, int rightY0, int rightY1, int topX0, int topX1) {
            this.bitmap = bitmap;
            this.maskWidth = maskWidth;
            this.maskHeight = maskHeight;
            this.leftY0 = leftY0;
            this.leftY1 = leftY1;
            this.rightY0 = rightY0;
            this.rightY1 = rightY1;
            this.topX0 = topX0;
            this.topX1 = topX1;
        }
    }

    /**
     * @param fadeFraction frame-crop fade width as a fraction of the
     *                     cut-out (>= EDGE_FADE_FRACTION; the still path
     *                     passes the placement-aware fraction from
     *                     edgeFadeFraction so a scaled-down person keeps
     *                     a perceptible ramp). Since v0.9.3 the fade
     *                     ALWAYS runs, in AI mode too: the seam band the
     *                     inpainter regenerates covers the fade zone, and
     *                     when the AI is off/unavailable/detects nothing,
     *                     the fade is what softens the frame-crop edge.
     */
    private Cutout buildCutout(Bitmap photo, MaskData mask, float fadeFraction) {
        int photoWidth = photo.getWidth();
        int photoHeight = photo.getHeight();
        int maskWidth = mask.width;
        int maskHeight = mask.height;
        int maskCount = maskWidth * maskHeight;
        if (photoWidth <= 0 || photoHeight <= 0 || maskCount <= 0) {
            return null;
        }
        // Work on a copy: boxBlur mutates the array in place, the cached
        // MaskData must stay pristine (presence scan reads it below).
        float[] conf = mask.conf.clone();
        // Feather the matte at capture resolution: the mask derives from a
        // ~256px model grid, so at full-res the raw transition is only a few
        // pixels wide and looks cut out. A blur radius proportional to the
        // photo size (~longSide/160) turns it into a soft, gradual edge.
        int featherPhotoPx = Math.max(2, Math.round(Math.max(photoWidth, photoHeight) / 160f));
        int featherRadius = Math.max(1,
                Math.round(featherPhotoPx * maskWidth / (float) photoWidth));
        boxBlur(conf, maskWidth, maskHeight, featherRadius);

        // M7 frame-crop presence: which borders the person silhouette was
        // CUT by (front camera frame), as min..max ranges in mask coords.
        // Scanned on the RAW mask confidence (mask.conf) since v0.9.3: the
        // feather blur smears a narrow cut strip below SEAM_PRESENCE_CONF,
        // which made the AI blend silently skip real seams on device.
        // -1/-1 = border not touched -> no seam band there.
        int leftY0 = -1, leftY1 = -1, rightY0 = -1, rightY1 = -1, topX0 = -1, topX1 = -1;
        for (int y = 0; y < maskHeight; y++) {
            int row = y * maskWidth;
            boolean left = false;
            boolean right = false;
            for (int k = 0; k < 3 && !(left && right); k++) {
                if (mask.conf[row + k] > SEAM_PRESENCE_CONF) {
                    left = true;
                }
                if (mask.conf[row + maskWidth - 1 - k] > SEAM_PRESENCE_CONF) {
                    right = true;
                }
            }
            if (left) {
                if (leftY0 < 0) leftY0 = y;
                leftY1 = y;
            }
            if (right) {
                if (rightY0 < 0) rightY0 = y;
                rightY1 = y;
            }
        }
        for (int x = 0; x < maskWidth; x++) {
            boolean top = false;
            for (int k = 0; k < 3 && !top; k++) {
                if (mask.conf[k * maskWidth + x] > SEAM_PRESENCE_CONF) {
                    top = true;
                }
            }
            if (top) {
                if (topX0 < 0) topX0 = x;
                topX1 = x;
            }
        }

        // The per-pixel compositing runs in native Skia, not in a Java loop
        // over every full-res pixel: quantize the blurred confidence to alpha
        // via the precomputed smoothstep LUT (one linear pass over the mask),
        // build an ARGB_8888 mask bitmap and apply it to the photo with a
        // DST_IN draw (SIMD-optimized). ARGB_8888 + DST_IN is the canonical
        // masking pattern; v0.6.1's ALPHA_8 + copyPixelsFromBuffer variant
        // silently produced an un-masked rectangle on the Pixel 9.
        //
        // Frame-crop fade (see EDGE_FADE_FRACTION, v0.8.1): the mask
        // confidence is multiplied by smoothstep ramps from the left/right
        // and top borders BEFORE the alpha quantization, so where the person
        // was cut by the front camera's frame the cut-out dissolves into the
        // back scene instead of showing a hard rectangular seam. The mask
        // maps 1:1 (or is scaled by the DST_IN draw) to the photo, so the
        // fade scales with the photo. The bottom stays hard (grounded
        // person, glued to the frame like a real photo crop).
        int[] maskArgb = new int[maskCount];
        int i = 0;
        int fadeW = Math.max(1, Math.round(maskWidth * fadeFraction));
        int fadeH = Math.max(1, Math.round(maskHeight * fadeFraction));
        float[] fadeX = new float[maskWidth];
        for (int x = 0; x < maskWidth; x++) {
            float fromLeft = x < fadeW ? smoothstep(x / (float) fadeW) : 1f;
            float fromRight = maskWidth - 1 - x < fadeW
                    ? smoothstep((maskWidth - 1 - x) / (float) fadeW) : 1f;
            fadeX[x] = Math.min(fromLeft, fromRight);
        }
        float[] fadeY = new float[maskHeight];
        for (int y = 0; y < maskHeight; y++) {
            fadeY[y] = y < fadeH ? smoothstep(y / (float) fadeH) : 1f;
        }
        for (int y = 0; y < maskHeight; y++) {
            float fy = fadeY[y];
            for (int x = 0; x < maskWidth; x++, i++) {
                int q = (int) (conf[i] * fy * fadeX[x] * 255f + 0.5f);
                if (q < 0) q = 0;
                else if (q > 255) q = 255;
                maskArgb[i] = (ALPHA_LUT[q] << 24) | 0x00FFFFFF;
            }
        }
        Bitmap alphaMask = Bitmap.createBitmap(maskArgb, maskWidth, maskHeight,
                Bitmap.Config.ARGB_8888);

        Bitmap cutout = Bitmap.createBitmap(photoWidth, photoHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(cutout);
        canvas.drawBitmap(photo, 0, 0, null);
        Paint maskPaint = new Paint();
        maskPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));
        boolean scaleMask = maskWidth != photoWidth || maskHeight != photoHeight;
        maskPaint.setFilterBitmap(scaleMask);
        RectF full = new RectF(0, 0, photoWidth, photoHeight);
        canvas.drawBitmap(alphaMask, null, full, maskPaint);
        return new Cutout(cutout, maskWidth, maskHeight,
                leftY0, leftY1, rightY0, rightY1, topX0, topX1);
    }

    /**
     * M7 seam-band geometry: full-resolution HOLE rectangles for the
     * inpainter, one per frame border that (a) the person silhouette was
     * cut by (presence ranges from buildCutout) and (b) lies INSIDE the
     * back photo - a cut flush with the photo border is a natural crop and
     * shows no seam. Each band straddles the cut-out's mapped edge
     * (inside half = person pixels, outside half = scene), sized from the
     * SAME fadeFraction the cut-out fade used (so the regenerated band
     * always covers the fade zone) and grown by SEAM_DILATE so the model
     * re-synthesizes the existing edge pixels too.
     *
     * v0.9.4 outpaint growth (LEFT/RIGHT only): the left/right holes grow
     * OUTWARD (into the background side) proportionally to the mask's
     * border-touch extent - a t-shirt cut along half the left border
     * gets a hole reaching ~15% of that extent into the scene, so the
     * model paints person continuation there instead of only softening
     * the seam strip. Top keeps the thin band (hair edges already read
     * well, user decision); the growth is axis-limited (OUTPAINT_MAX_AXIS_PX)
     * so the single 512px model window never drops below ~448/1600
     * resolution (worse than today's long bands = bubbly texture).
     */
    private static ArrayList<Rect> buildSeamHoleRects(int fusedW, int fusedH,
                                                      Matrix placement, int cutW, int cutH,
                                                      Cutout cut, float fadeFraction) {
        float[] v = new float[9];
        placement.getValues(v);
        float scale = v[Matrix.MSCALE_X];
        int cx0 = Math.round(v[Matrix.MTRANS_X]);
        int cy0 = Math.round(v[Matrix.MTRANS_Y]);
        int cx1 = Math.round(v[Matrix.MTRANS_X] + cutW * scale);
        int cy1 = Math.round(v[Matrix.MTRANS_Y] + cutH * scale);
        int bandHalf = Math.round(cutW * scale * fadeFraction);
        if (bandHalf < 24) {
            bandHalf = 24;
        } else if (bandHalf > 160) {
            bandHalf = 160;
        }
        ArrayList<Rect> rects = new ArrayList<>();
        if (cut.leftY1 >= 0 && cx0 > 0) {
            int y0 = mapToCut(cut.leftY0, cut.maskHeight, cy0, cy1);
            int y1 = mapToCut(cut.leftY1, cut.maskHeight, cy0, cy1);
            addClipped(rects,
                    cx0 - bandHalf - SEAM_DILATE - outGrow(y1 - y0, bandHalf),
                    y0 - SEAM_DILATE,
                    cx0 + bandHalf + SEAM_DILATE,
                    y1 + SEAM_DILATE,
                    fusedW, fusedH);
        }
        if (cut.rightY1 >= 0 && cx1 < fusedW) {
            int y0 = mapToCut(cut.rightY0, cut.maskHeight, cy0, cy1);
            int y1 = mapToCut(cut.rightY1, cut.maskHeight, cy0, cy1);
            addClipped(rects,
                    cx1 - bandHalf - SEAM_DILATE,
                    y0 - SEAM_DILATE,
                    cx1 + bandHalf + SEAM_DILATE + outGrow(y1 - y0, bandHalf),
                    y1 + SEAM_DILATE,
                    fusedW, fusedH);
        }
        if (cut.topX1 >= 0 && cy0 > 0) {
            addClipped(rects,
                    mapToCut(cut.topX0, cut.maskWidth, cx0, cx1) - SEAM_DILATE,
                    cy0 - bandHalf - SEAM_DILATE,
                    mapToCut(cut.topX1, cut.maskWidth, cx0, cx1) + SEAM_DILATE,
                    cy0 + bandHalf + SEAM_DILATE,
                    fusedW, fusedH);
        }
        return rects;
    }

    /**
     * v0.9.4 outpaint hole growth for one left/right seam band: the
     * outward reach into the background, proportional to how much of
     * the frame border the person mask touches (the "intersection").
     * Floored so even a small cut gets a continuation worth calling
     * outpainting, capped in absolute px, and axis-limited so the
     * grown band's cross axis never exceeds OUTPAINT_MAX_AXIS_PX (the
     * 512px model window then never scales below ~448/1600).
     */
    private static int outGrow(int touchLen, int bandHalf) {
        if (touchLen <= 0) {
            return 0;
        }
        int grow = Math.round(touchLen * OUTPAINT_GROW_FRACTION);
        if (grow < OUTPAINT_MIN_EXTEND_PX) {
            grow = OUTPAINT_MIN_EXTEND_PX;
        } else if (grow > OUTPAINT_GROW_MAX_PX) {
            grow = OUTPAINT_GROW_MAX_PX;
        }
        // Axis limit: cross axis = band (2*(bandHalf+SEAM_DILATE)) + grow.
        int maxGrow = OUTPAINT_MAX_AXIS_PX - 2 * (bandHalf + SEAM_DILATE);
        if (grow > maxGrow) {
            grow = maxGrow;
        }
        return Math.max(0, grow);
    }

    /** Maps a mask coordinate to the fused photo (mask -> cut-out -> placement). */
    private static int mapToCut(int m, int maskDim, int c0, int c1) {
        return Math.round(c0 + (m / (float) maskDim) * (c1 - c0));
    }

    private static void addClipped(ArrayList<Rect> rects, int l, int t, int r, int b,
                                   int w, int h) {
        if (l < 0) l = 0;
        if (t < 0) t = 0;
        if (r > w) r = w;
        if (b > h) b = h;
        if (r - l >= 8 && b - t >= 8) {
            rects.add(new Rect(l, t, r, b));
        }
    }

    /**
     * M3 color match: transfers the back scene's lighting statistics onto
     * the cut-out person (Reinhard-style statistical color transfer,
     * computed in a luma/chroma split of sRGB: Y = Rec.709 luma, U = R-Y,
     * V = B-Y):
     *   - luma (v0.9.2 rebalance): the contrast (std) ratio is clamped and
     *     DAMPED toward 1, and the mean moves only partially toward the
     *     scene's plus a subject-brighten offset - full adoption darkened
     *     the person in practice (front cams expose faces brighter than
     *     the scene average); see the M3 tunables comment,
     *   - chroma: the mean U/V delta - the white-balance gap between the
     *     two cameras - is applied damped and capped.
     * The person statistics are weighted by the cut-out's own alpha, so
     * only person pixels contribute; the scene statistics come from the
     * whole back photo (global illuminant estimate). The complete
     * transfer is affine in RGB, so it is applied with ONE native
     * ColorMatrix draw instead of a full-res Java pixel loop (same
     * philosophy as the v0.6.1/v0.7.0 compositing rework). Returns the
     * matched bitmap (input recycled); on degenerate statistics the
     * input is returned unchanged.
     */
    private Bitmap colorMatchToScene(Bitmap cutout, Bitmap scene) {
        float[] person = new float[4];   // meanY, stdY, meanU, meanV
        float[] target = new float[4];
        if (!cutoutStats(cutout, person) || !sceneStats(scene, target)) {
            return cutout;
        }
        float kY = target[1] / Math.max(person[1], 0.001f);
        if (kY < COLOR_LUMA_SCALE_MIN) {
            kY = COLOR_LUMA_SCALE_MIN;
        } else if (kY > COLOR_LUMA_SCALE_MAX) {
            kY = COLOR_LUMA_SCALE_MAX;
        }
        // v0.9.2: contrast adoption damped toward 1 - full adoption
        // stretched the person's midtones when the scene's std was much
        // larger and crushed them dark via the mean re-center.
        kY = 1f + (kY - 1f) * COLOR_LUMA_CONTRAST_STRENGTH;
        if (person[1] < 2f || target[1] < 2f) {
            // Essentially flat signal: the contrast ratio is meaningless.
            kY = 1f;
        }
        float dU = (target[2] - person[2]) * COLOR_CHROMA_STRENGTH;
        float dV = (target[3] - person[3]) * COLOR_CHROMA_STRENGTH;
        float shiftU = Math.max(-COLOR_CHROMA_SHIFT_MAX,
                Math.min(COLOR_CHROMA_SHIFT_MAX, dU));
        float shiftV = Math.max(-COLOR_CHROMA_SHIFT_MAX,
                Math.min(COLOR_CHROMA_SHIFT_MAX, dV));

        // v0.9.2: mean adoption damped + subject emphasis (see the M3
        // tunables comment) - move only COLOR_LUMA_MEAN_STRENGTH of the
        // person->scene luma gap and stay COLOR_SUBJECT_BRIGHTEN above
        // it, so the person adopts the scene's tonal direction without
        // being dragged down to the scene average.
        float anchor = person[0] + (target[0] - person[0]) * COLOR_LUMA_MEAN_STRENGTH
                + COLOR_SUBJECT_BRIGHTEN;
        // Transfer in (Y, U, V): Y2 = kY*Y + cy, U2 = U + shiftU,
        // V2 = V + shiftV, then back to RGB via R = Y + U, B = Y + V,
        // G = Y - (wR*U + wB*V)/wG. Every output channel is linear in
        // R, G, B plus a constant, i.e. exactly a ColorMatrix.
        float cy = anchor - kY * person[0];
        float ky1 = kY - 1f;
        float gk = kY + (LUMA_R + LUMA_B) / LUMA_G;
        float tR = cy + shiftU;
        float tB = cy + shiftV;
        float tG = cy - (LUMA_R * shiftU + LUMA_B * shiftV) / LUMA_G;
        ColorMatrix matrix = new ColorMatrix(new float[]{
                ky1 * LUMA_R + 1f, ky1 * LUMA_G, ky1 * LUMA_B, 0, tR,
                gk * LUMA_R - LUMA_R / LUMA_G, gk * LUMA_G,
                gk * LUMA_B - LUMA_B / LUMA_G, 0, tG,
                ky1 * LUMA_R, ky1 * LUMA_G, ky1 * LUMA_B + 1f, 0, tB,
                0, 0, 0, 1, 0});
        Bitmap matched = Bitmap.createBitmap(cutout.getWidth(), cutout.getHeight(),
                Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(matched);
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setColorFilter(new ColorMatrixColorFilter(matrix));
        canvas.drawBitmap(cutout, 0, 0, paint);
        cutout.recycle();
        return matched;
    }

    /** Alpha-weighted luma/chroma statistics of the cut-out (person only). */
    private static boolean cutoutStats(Bitmap cutout, float[] out) {
        Bitmap small = downscaleForStats(cutout);
        try {
            int w = small.getWidth();
            int h = small.getHeight();
            int[] px = new int[w * h];
            small.getPixels(px, 0, w, 0, 0, w, h);
            double sw = 0, sy = 0, syy = 0, su = 0, sv = 0;
            for (int p : px) {
                int a = (p >>> 24);
                if (a == 0) {
                    continue;
                }
                double wt = a / 255.0;
                double r = (p >> 16) & 0xFF;
                double g = (p >> 8) & 0xFF;
                double b = p & 0xFF;
                double y = LUMA_R * r + LUMA_G * g + LUMA_B * b;
                sw += wt;
                sy += wt * y;
                syy += wt * y * y;
                su += wt * (r - y);
                sv += wt * (b - y);
            }
            if (sw < 1.0) {
                return false;
            }
            double meanY = sy / sw;
            out[0] = (float) meanY;
            out[1] = (float) Math.sqrt(Math.max(0.0, syy / sw - meanY * meanY));
            out[2] = (float) (su / sw);
            out[3] = (float) (sv / sw);
            return true;
        } finally {
            if (small != cutout) {
                small.recycle();
            }
        }
    }

    /** Unweighted luma/chroma statistics of the back scene. */
    private static boolean sceneStats(Bitmap scene, float[] out) {
        Bitmap small = downscaleForStats(scene);
        try {
            int w = small.getWidth();
            int h = small.getHeight();
            int[] px = new int[w * h];
            small.getPixels(px, 0, w, 0, 0, w, h);
            double n = 0, sy = 0, syy = 0, su = 0, sv = 0;
            for (int p : px) {
                double r = (p >> 16) & 0xFF;
                double g = (p >> 8) & 0xFF;
                double b = p & 0xFF;
                double y = LUMA_R * r + LUMA_G * g + LUMA_B * b;
                n += 1;
                sy += y;
                syy += y * y;
                su += r - y;
                sv += b - y;
            }
            if (n < 1.0) {
                return false;
            }
            double meanY = sy / n;
            out[0] = (float) meanY;
            out[1] = (float) Math.sqrt(Math.max(0.0, syy / n - meanY * meanY));
            out[2] = (float) (su / n);
            out[3] = (float) (sv / n);
            return true;
        } finally {
            if (small != scene) {
                small.recycle();
            }
        }
    }

    private static Bitmap downscaleForStats(Bitmap src) {
        int longSide = Math.max(src.getWidth(), src.getHeight());
        if (longSide <= COLOR_STATS_SIDE) {
            return src;
        }
        float scale = COLOR_STATS_SIDE / (float) longSide;
        int w = Math.max(1, Math.round(src.getWidth() * scale));
        int h = Math.max(1, Math.round(src.getHeight() * scale));
        return Bitmap.createScaledBitmap(src, w, h, true);
    }

    private Bitmap upright(byte[] jpeg, int rotation, boolean mirror) {
        Bitmap bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        if (bitmap == null) {
            return null;
        }
        if (rotation == 0 && !mirror) {
            return bitmap;
        }
        Matrix matrix = new Matrix();
        if (rotation != 0) {
            matrix.postRotate(rotation);
        }
        if (mirror) {
            matrix.postScale(-1f, 1f);
        }
        Bitmap out = Bitmap.createBitmap(bitmap, 0, 0,
                bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        if (out != bitmap) {
            bitmap.recycle();
        }
        return out;
    }

    private void saveResult(byte[] originalJpeg, Bitmap photo, SegmentationMask mask) {
        try {
            showCaptureStage("Saving...");
            Cutout result = buildCutout(photo, readMask(mask), EDGE_FADE_FRACTION);
            Bitmap cutout = result == null ? null : result.bitmap;
            photo.recycle();
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            if (saveExtras) {
                saveJpeg(originalJpeg, "SF_original_" + stamp + ".jpg");
            }
            if (cutout != null) {
                Uri cutoutUri = saveBitmap(cutout, "SF_cutout_" + stamp + ".png", "image/png", true);
                cutout.recycle();
                openSavedImagePreview(cutoutUri, "image/png");
            }
            finishCaptureUi(true, "Saved to Pictures/SelfieFusion");
        } catch (OutOfMemoryError e) {
            finishCaptureUi(false, null);
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Out of memory processing still", Toast.LENGTH_LONG).show());
        } catch (Throwable t) {
            finishCaptureUi(false, null);
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Save error: " + t.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    /**
     * Lazily loads the APK-bundled LaMa inpainting model (M7) - the
     * first call extracts it from the APK asset to the files dir. The
     * session stays loaded for subsequent captures; a failed attempt is
     * remembered so the save pipeline stops trying (every capture then
     * uses the edge fade fallback without further cost).
     */
    private Inpainter ensureInpainter() {
        if (inpainter != null) {
            return inpainter;
        }
        if (inpainterUnavailable) {
            return null;
        }
        inpainter = Inpainter.create(this);
        if (inpainter == null) {
            inpainterUnavailable = true;
            Log.w(TAG, "inpainting model unusable - seam blend falls back to edge fade");
        }
        return inpainter;
    }

    private void saveJpeg(byte[] jpeg, String name) {
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                ContentValues values = new ContentValues();
                values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
                values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                values.put(MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_PICTURES + "/SelfieFusion");
                values.put(MediaStore.Images.Media.IS_PENDING, 1);
                ContentResolver resolver = getContentResolver();
                Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                if (uri == null) {
                    return;
                }
                try (OutputStream os = resolver.openOutputStream(uri)) {
                    os.write(jpeg);
                }
                values.clear();
                values.put(MediaStore.Images.Media.IS_PENDING, 0);
                resolver.update(uri, values, null, null);
            } else {
                File dir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "SelfieFusion");
                if (!dir.exists() && !dir.mkdirs()) {
                    return;
                }
                try (FileOutputStream fos = new FileOutputStream(new File(dir, name))) {
                    fos.write(jpeg);
                }
            }
        } catch (Throwable t) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Save failed: " + t.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    private Uri saveBitmap(Bitmap bitmap, String name, String mimeType, boolean isPng) {
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                ContentValues values = new ContentValues();
                values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
                values.put(MediaStore.Images.Media.MIME_TYPE, mimeType);
                values.put(MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_PICTURES + "/SelfieFusion");
                values.put(MediaStore.Images.Media.IS_PENDING, 1);
                ContentResolver resolver = getContentResolver();
                Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                if (uri == null) {
                    return null;
                }
                try (OutputStream os = resolver.openOutputStream(uri)) {
                    if (isPng) {
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
                    } else {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, os);
                    }
                }
                values.clear();
                values.put(MediaStore.Images.Media.IS_PENDING, 0);
                resolver.update(uri, values, null, null);
                return uri;
            } else {
                File dir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "SelfieFusion");
                if (!dir.exists() && !dir.mkdirs()) {
                    return null;
                }
                File file = new File(dir, name);
                try (OutputStream os = new FileOutputStream(file)) {
                    if (isPng) {
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
                    } else {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, os);
                    }
                }
                return FileProvider.getUriForFile(this,
                        "com.selfiefusion.segtest.fileprovider", file);
            }
        } catch (Throwable t) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Save failed: " + t.getMessage(), Toast.LENGTH_LONG).show());
            return null;
        }
    }

    // Saved-still preview: ACTION_VIEW on the saved Uri only - the
    // in-memory bitmap is already recycled by the time the viewer opens.
    private void openSavedImagePreview(Uri uri, String mimeType) {
        if (uri == null) {
            Log.i(TAG, "image preview skipped: nothing saved");
            return;
        }
        mainHandler.post(() -> {
            try {
                Intent intent = new Intent(Intent.ACTION_VIEW);
                intent.setDataAndType(uri, mimeType);
                intent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                startActivity(intent);
            } catch (ActivityNotFoundException e) {
                Toast.makeText(MainActivity.this, "No image viewer app found",
                        Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mainHandler.removeCallbacks(fpsRunnable);
        mainHandler.removeCallbacks(hideDoneCard);
        if (streamSegmenter != null) {
            streamSegmenter.close();
        }
        if (stillSegmenter != null) {
            stillSegmenter.close();
        }
        if (inpainter != null) {
            inpainter.close();
            inpainter = null;
        }
        cameraExecutor.shutdown();
        composeExecutor.shutdown();
    }
}