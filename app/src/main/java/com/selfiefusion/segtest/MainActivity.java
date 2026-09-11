package com.selfiefusion.segtest;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
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
import android.widget.PopupMenu;
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

    private PreviewView previewView;
    private PreviewView backPreviewView;
    private MaskOverlayView overlayView;
    private TextView infoView;
    private Button shutterButton;
    private ImageButton modeButton;

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

    private boolean saveExtras = false;
    private boolean showStats = false;

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
            for (int x = 0; x < tw; x++) {
                int q = (int) (conf[maskRow + maskXOfs[x]] * 255f + 0.5f);
                if (q < 0) q = 0;
                else if (q > 255) q = 255;
                int i = row + x;
                personPixels[i] = (ALPHA_LUT[q] << 24) | (personPixels[i] & 0x00FFFFFF);
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
        menu.findItem(R.id.show_stats).setChecked(showStats);
        popup.setOnMenuItemClickListener(item -> {
            int id = item.getItemId();
            if (id == R.id.save_extras) {
                saveExtras = !saveExtras;
                item.setChecked(saveExtras);
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
        String shutterHint;
        String extrasNote = saveExtras ? " (+ originals)" : "";
        if (dualActive && frontCaptureBound && backCaptureBound) {
            shutterHint = "Shutter: save FUSED selfie JPG" + extrasNote;
        } else if (frontCaptureBound) {
            shutterHint = (dualActive ? "Shutter: front cut-out PNG only (no back still)"
                    : "Shutter: save cut-out PNG") + extrasNote;
        } else {
            shutterHint = "Shutter: disabled in this camera mode";
        }
        String text = String.format(Locale.US,
                "%s | %.0f FPS | mask %dx%d %s | rot %d | %s cam\n"
                        + "Menu (top-right): view mode | drag/pinch person (Composite) | %s",
                MODE_NAMES[overlayView.getMode()], lastFps,
                overlayView.getMaskWidth(), overlayView.getMaskHeight(),
                overlayView.getMaskFormat(), lastRotation, cameraLabel, shutterHint);
        infoView.setText(text);
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
        Toast.makeText(this, "Capturing + segmenting full-res...", Toast.LENGTH_SHORT).show();
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
                        runOnUiThread(() -> Toast.makeText(MainActivity.this,
                                "Failed to decode captured image", Toast.LENGTH_LONG).show());
                        return;
                    }
                    stillSegmenter.process(InputImage.fromBitmap(photo, 0))
                            .addOnSuccessListener(cameraExecutor, mask -> saveResult(jpeg, photo, mask))
                            .addOnFailureListener(cameraExecutor, e -> {
                                photo.recycle();
                                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                                        "Segmentation failed: " + e.getMessage(), Toast.LENGTH_LONG).show());
                            })
                            .addOnCompleteListener(cameraExecutor, r -> capturing = false);
                } catch (Throwable t) {
                    capturing = false;
                    runOnUiThread(() -> Toast.makeText(MainActivity.this,
                            "Capture error: " + t.getMessage(), Toast.LENGTH_LONG).show());
                }
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                capturing = false;
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
        Toast.makeText(this, "Capturing both cameras + fusing...", Toast.LENGTH_SHORT).show();
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
                return;
            }
            stillSegmenter.process(InputImage.fromBitmap(front, 0))
                    .addOnSuccessListener(cameraExecutor, mask ->
                            processFused(front, frontJpeg, backJpeg, backRotation, mask,
                                    livePlacement))
                    .addOnFailureListener(cameraExecutor, e -> {
                        front.recycle();
                        runOnUiThread(() -> Toast.makeText(MainActivity.this,
                                "Segmentation failed: " + e.getMessage(), Toast.LENGTH_LONG).show());
                        capturing = false;
                    });
        } catch (Throwable t) {
            capturing = false;
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Fusion error: " + t.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

private void processFused(Bitmap front, byte[] frontJpeg, byte[] backJpeg,
                              int backRotation, SegmentationMask mask,
                              MaskOverlayView.PersonPlacement placement) {
        Bitmap cutout = null;
        Bitmap back = null;
        Bitmap fused = null;
        try {
            cutout = buildCutout(front, mask);
            front.recycle();
            if (cutout == null) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Segmentation mask invalid", Toast.LENGTH_LONG).show());
                return;
            }
            back = upright(backJpeg, backRotation, false);
            if (back == null) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Failed to decode back image", Toast.LENGTH_LONG).show());
                return;
            }
            fused = Bitmap.createBitmap(back.getWidth(), back.getHeight(), Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(fused);
            canvas.drawBitmap(back, 0, 0, null);
            Matrix matrix = placementForCutout(cutout, back, placement);
            Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
            canvas.drawBitmap(cutout, matrix, paint);

            if (saveExtras) {
                saveJpeg(frontJpeg, "SF_front_" + dualStamp + ".jpg");
                saveJpeg(backJpeg, "SF_back_" + dualStamp + ".jpg");
                saveBitmap(cutout, "SF_cutout_" + dualStamp + ".png", "image/png", true);
            }
            saveBitmap(fused, "SF_fused_" + dualStamp + ".jpg", "image/jpeg", false);
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    saveExtras ? "Fused selfie + materials saved to Pictures/SelfieFusion"
                            : "Fused selfie saved to Pictures/SelfieFusion",
                    Toast.LENGTH_LONG).show());
        } catch (OutOfMemoryError e) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Out of memory processing fused still", Toast.LENGTH_LONG).show());
        } catch (Throwable t) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Fusion error: " + t.getMessage(), Toast.LENGTH_LONG).show());
        } finally {
            if (cutout != null) {
                cutout.recycle();
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

    /**
     * Transform that reproduces the on-screen person placement inside the
     * saved back photo. The back preview (PreviewView FILL_CENTER) shows a
     * centered crop of the back photo, so view coordinates map linearly
     * into photo coordinates; the person height arranged on screen is kept
     * and the bottom edge is glued to the bottom of the photo (grounded
     * person, never floating).
     */
    private static Matrix placementForCutout(Bitmap cutout, Bitmap back,
                                             MaskOverlayView.PersonPlacement placement) {
        int backW = back.getWidth();
        int backH = back.getHeight();
        Matrix matrix = new Matrix();
        if (placement == null || !placement.valid) {
            // Fallback: cover the back photo, bottom edge glued to the frame.
            float scale = Math.max(backW / (float) cutout.getWidth(),
                    backH / (float) cutout.getHeight());
            matrix.postScale(scale, scale);
            matrix.postTranslate((backW - cutout.getWidth() * scale) / 2f,
                    backH - cutout.getHeight() * scale);
            return matrix;
        }
        float fit = Math.max(placement.viewWidth / (float) placement.bitmapWidth,
                placement.viewHeight / (float) placement.bitmapHeight);
        float personHView = placement.bitmapHeight * fit * placement.scale;
        float viewToBack = 1f / Math.max(placement.viewWidth / (float) backW,
                placement.viewHeight / (float) backH);
        float scale = personHView * viewToBack / cutout.getHeight();
        float centerXPhoto = (placement.centerX - placement.viewWidth / 2f) * viewToBack
                + backW / 2f;
        float left = centerXPhoto - cutout.getWidth() * scale / 2f;
        float top = backH - cutout.getHeight() * scale;
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

    private Bitmap buildCutout(Bitmap photo, SegmentationMask mask) {
        int photoWidth = photo.getWidth();
        int photoHeight = photo.getHeight();
        ByteBuffer buffer = mask.getBuffer();
        int maskWidth = mask.getWidth();
        int maskHeight = mask.getHeight();
        int maskCount = maskWidth * maskHeight;
        if (photoWidth <= 0 || photoHeight <= 0 || maskCount <= 0) {
            return null;
        }
        boolean floatMask = buffer.remaining() >= maskCount * 4L;
        float[] conf = new float[maskCount];
        if (floatMask) {
            buffer.asFloatBuffer().get(conf, 0, maskCount);
        } else {
            for (int i = 0; i < maskCount; i++) {
                conf[i] = (buffer.get() & 0xFF) / 255f;
            }
        }
        // Feather the matte at capture resolution: the mask derives from a
        // ~256px model grid, so at full-res the raw transition is only a few
        // pixels wide and looks cut out. A blur radius proportional to the
        // photo size (~longSide/160) turns it into a soft, gradual edge.
        int featherPhotoPx = Math.max(2, Math.round(Math.max(photoWidth, photoHeight) / 160f));
        int featherRadius = Math.max(1,
                Math.round(featherPhotoPx * maskWidth / (float) photoWidth));
        boxBlur(conf, maskWidth, maskHeight, featherRadius);

        // The per-pixel compositing runs in native Skia, not in a Java loop
        // over every full-res pixel: quantize the blurred confidence to alpha
        // via the precomputed smoothstep LUT (one linear pass over the mask),
        // build an ARGB_8888 mask bitmap and apply it to the photo with a
        // DST_IN draw (SIMD-optimized). ARGB_8888 + DST_IN is the canonical
        // masking pattern; v0.6.1's ALPHA_8 + copyPixelsFromBuffer variant
        // silently produced an un-masked rectangle on the Pixel 9.
        int[] maskArgb = new int[maskCount];
        for (int i = 0; i < maskCount; i++) {
            int q = (int) (conf[i] * 255f + 0.5f);
            if (q < 0) q = 0;
            else if (q > 255) q = 255;
            maskArgb[i] = (ALPHA_LUT[q] << 24) | 0x00FFFFFF;
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
        return cutout;
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
            Bitmap cutout = buildCutout(photo, mask);
            photo.recycle();
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            if (saveExtras) {
                saveJpeg(originalJpeg, "SF_original_" + stamp + ".jpg");
            }
            if (cutout != null) {
                saveBitmap(cutout, "SF_cutout_" + stamp + ".png", "image/png", true);
                cutout.recycle();
            }
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Saved to Pictures/SelfieFusion", Toast.LENGTH_LONG).show());
        } catch (OutOfMemoryError e) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Out of memory processing still", Toast.LENGTH_LONG).show());
        } catch (Throwable t) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Save error: " + t.getMessage(), Toast.LENGTH_LONG).show());
        }
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

    private void saveBitmap(Bitmap bitmap, String name, String mimeType, boolean isPng) {
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
                    return;
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
            } else {
                File dir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "SelfieFusion");
                if (!dir.exists() && !dir.mkdirs()) {
                    return;
                }
                try (OutputStream os = new FileOutputStream(new File(dir, name))) {
                    if (isPng) {
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
                    } else {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, os);
                    }
                }
            }
        } catch (Throwable t) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this,
                    "Save failed: " + t.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mainHandler.removeCallbacks(fpsRunnable);
        if (streamSegmenter != null) {
            streamSegmenter.close();
        }
        if (stillSegmenter != null) {
            stillSegmenter.close();
        }
        cameraExecutor.shutdown();
        composeExecutor.shutdown();
    }
}