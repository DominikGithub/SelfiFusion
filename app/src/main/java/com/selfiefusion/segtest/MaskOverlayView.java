package com.selfiefusion.segtest;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;

public class MaskOverlayView extends View {

    public static final int MODE_COMPOSITE = 0;
    public static final int MODE_RED_BG = 1;
    public static final int MODE_HEATMAP = 2;
    public static final int MODE_OFF = 3;

    private static final float MIN_PERSON_SCALE = 0.25f;
    private static final float MAX_PERSON_SCALE = 5f;

    public static class PersonPlacement {
        public final boolean valid;
        public final int viewWidth;
        public final int viewHeight;
        public final int bitmapWidth;
        public final int bitmapHeight;
        public final float centerX;
        public final float scale;

        PersonPlacement(boolean valid, int viewWidth, int viewHeight,
                        int bitmapWidth, int bitmapHeight, float centerX, float scale) {
            this.valid = valid;
            this.viewWidth = viewWidth;
            this.viewHeight = viewHeight;
            this.bitmapWidth = bitmapWidth;
            this.bitmapHeight = bitmapHeight;
            this.centerX = centerX;
            this.scale = scale;
        }
    }

    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final RectF dst = new RectF();
    private final Matrix personMatrix = new Matrix();

    private int maskWidth;
    private int maskHeight;
    private volatile int mode = MODE_COMPOSITE;
    private boolean mirrored = true;
    private String maskFormat = "-";
    private volatile boolean dualActive = false;
    private Bitmap personBitmap;

    private boolean personTransformed = false;
    private float personScale = 1f;
    private float personCenterX;

    private boolean gesturePinching = false;
    private float lastTouchX;
    private float lastTouchY;
    private float lastPinchDist;
    private float lastPinchMidX;
    private float downX;
    private float downY;
    private boolean movedBeyondSlop = false;
    private long lastTapTime = 0;
    private float lastTapX;
    private float lastTapY;
    private final float touchSlop = ViewConfiguration.get(getContext()).getScaledTouchSlop();

    private Bitmap bitmap;

    public MaskOverlayView(Context context) {
        super(context);
    }

    public MaskOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public MaskOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public void setMode(int newMode) {
        mode = newMode;
        postInvalidate();
    }

    public int getMode() {
        return mode;
    }

    public void setDualActive(boolean value) {
        dualActive = value;
        if (!dualActive && mode == MODE_COMPOSITE) {
            mode = MODE_RED_BG;
        }
        postInvalidate();
    }

    public int getMaskWidth() {
        return maskWidth;
    }

    public int getMaskHeight() {
        return maskHeight;
    }

    public String getMaskFormat() {
        return maskFormat;
    }

    public void setMirrored(boolean value) {
        mirrored = value;
    }

    public void updateFrame(int[] maskPixels, int width, int height, String format, Bitmap person) {
        maskWidth = width;
        maskHeight = height;
        maskFormat = format;
        if (maskPixels != null && maskPixels.length == width * height) {
            if (bitmap == null || bitmap.getWidth() != width || bitmap.getHeight() != height) {
                if (bitmap != null) {
                    bitmap.recycle();
                }
                bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            }
            bitmap.setPixels(maskPixels, 0, width, 0, 0, width, height);
        }
        if (dualActive) {
            personBitmap = person;
        }
        postInvalidate();
    }

    public void resetPersonTransform() {
        personTransformed = false;
        personScale = 1f;
        postInvalidate();
    }

    /**
     * Snapshot of the live person placement (call from the UI thread, e.g.
     * at shutter press) so the saved still can reproduce exactly what the
     * user arranged on screen.
     */
    public PersonPlacement snapshotPlacement() {
        if (!dualActive || personBitmap == null) {
            return new PersonPlacement(false, 0, 0, 0, 0, 0, 1);
        }
        int viewWidth = getWidth();
        int viewHeight = getHeight();
        if (viewWidth <= 0 || viewHeight <= 0) {
            return new PersonPlacement(false, 0, 0, 0, 0, 0, 1);
        }
        return new PersonPlacement(true, viewWidth, viewHeight,
                personBitmap.getWidth(), personBitmap.getHeight(),
                personTransformed ? personCenterX : viewWidth / 2f,
                personTransformed ? personScale : 1f);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (mode != MODE_COMPOSITE || !dualActive || personBitmap == null) {
            return false;
        }
        int viewWidth = getWidth();
        int viewHeight = getHeight();
        if (viewWidth <= 0 || viewHeight <= 0) {
            return false;
        }
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                downX = event.getX();
                downY = event.getY();
                lastTouchX = downX;
                lastTouchY = downY;
                movedBeyondSlop = false;
                gesturePinching = false;
                return true;

            case MotionEvent.ACTION_POINTER_DOWN:
                if (event.getPointerCount() == 2) {
                    gesturePinching = true;
                    lastPinchDist = pinchDistance(event);
                    lastPinchMidX = pinchMidX(event);
                }
                return true;

            case MotionEvent.ACTION_MOVE:
                if (!movedBeyondSlop) {
                    float dx = event.getX() - downX;
                    float dy = event.getY() - downY;
                    if (dx * dx + dy * dy > touchSlop * touchSlop) {
                        movedBeyondSlop = true;
                        ensurePersonTransform(viewWidth, viewHeight);
                    }
                }
                if (movedBeyondSlop) {
                    if (gesturePinching && event.getPointerCount() >= 2) {
                        applyPinch(event, viewWidth, viewHeight);
                    } else if (event.getPointerCount() == 1) {
                        float x = event.getX();
                        float y = event.getY();
                        personCenterX = clamp(personCenterX + x - lastTouchX, 0, viewWidth);
                        // The person is grounded: its bottom edge stays glued
                        // to the bottom of the frame, so a vertical drag
                        // changes the size instead of lifting it up.
                        personScale = clamp(
                                personScale * (1f - (y - lastTouchY) / personDrawHeight(viewWidth, viewHeight)),
                                MIN_PERSON_SCALE, MAX_PERSON_SCALE);
                    }
                }
                lastTouchX = event.getX();
                lastTouchY = event.getY();
                postInvalidate();
                return true;

            case MotionEvent.ACTION_POINTER_UP:
                if (event.getPointerCount() == 2) {
                    gesturePinching = false;
                    int remaining = (event.getActionIndex() == 0) ? 1 : 0;
                    lastTouchX = event.getX(remaining);
                    lastTouchY = event.getY(remaining);
                }
                return true;

            case MotionEvent.ACTION_UP:
                if (!movedBeyondSlop) {
                    long now = event.getEventTime();
                    float tapDx = downX - lastTapX;
                    float tapDy = downY - lastTapY;
                    if (lastTapTime > 0
                            && now - lastTapTime < ViewConfiguration.getDoubleTapTimeout()
                            && tapDx * tapDx + tapDy * tapDy < touchSlop * touchSlop * 4f) {
                        resetPersonTransform();
                        lastTapTime = 0;
                    } else {
                        lastTapTime = now;
                        lastTapX = downX;
                        lastTapY = downY;
                    }
                }
                gesturePinching = false;
                return true;

            case MotionEvent.ACTION_CANCEL:
                gesturePinching = false;
                return true;

            default:
                return true;
        }
    }

    private void applyPinch(MotionEvent event, int viewWidth, int viewHeight) {
        float dist = pinchDistance(event);
        if (lastPinchDist <= 0f || dist <= 0f) {
            lastPinchDist = dist;
            return;
        }
        float oldScale = personScale;
        float candidate = clamp(oldScale * (dist / lastPinchDist),
                MIN_PERSON_SCALE, MAX_PERSON_SCALE);
        float kEff = candidate / oldScale;
        personScale = candidate;
        // Pinch keeps the horizontal anchor under the fingers but the
        // bottom edge stays glued to the frame (grounded person).
        float midX = pinchMidX(event);
        personCenterX = clamp(midX + (personCenterX - midX) * kEff
                + midX - lastPinchMidX, 0, viewWidth);
        lastPinchDist = dist;
        lastPinchMidX = midX;
    }

    private float pinchDistance(MotionEvent event) {
        float dx = event.getX(0) - event.getX(1);
        float dy = event.getY(0) - event.getY(1);
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    private float pinchMidX(MotionEvent event) {
        return (event.getX(0) + event.getX(1)) / 2f;
    }

    private void ensurePersonTransform(int viewWidth, int viewHeight) {
        if (!personTransformed) {
            personTransformed = true;
            personCenterX = viewWidth / 2f;
            personScale = 1f;
        }
    }

    private static float clamp(float value, float min, float max) {
        if (value < min) {
            return min;
        }
        if (value > max) {
            return max;
        }
        return value;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (mode == MODE_OFF) {
            return;
        }
        int viewWidth = getWidth();
        int viewHeight = getHeight();
        if (viewWidth <= 0 || viewHeight <= 0) {
            return;
        }
        if (mode == MODE_COMPOSITE) {
            if (personBitmap != null) {
                drawPerson(canvas, viewWidth, viewHeight);
            }
            return;
        }
        if (bitmap != null) {
            drawStretched(canvas, bitmap, viewWidth, viewHeight);
        }
    }

    private void drawPerson(Canvas canvas, int viewWidth, int viewHeight) {
        float fit = Math.max(viewWidth / (float) personBitmap.getWidth(),
                viewHeight / (float) personBitmap.getHeight());
        float s = fit * personScale;
        float cx = personTransformed ? personCenterX : viewWidth / 2f;
        float w = personBitmap.getWidth() * s;
        float h = personBitmap.getHeight() * s;
        // Grounded person: bottom edge flush with the bottom of the frame,
        // like a person cropped by the photo border in a real picture.
        float cy = viewHeight - h / 2f;
        personMatrix.reset();
        personMatrix.postScale(-s, s);
        personMatrix.postTranslate(cx + w / 2f, cy - h / 2f);
        canvas.drawBitmap(personBitmap, personMatrix, paint);
    }

    private float personDrawHeight(int viewWidth, int viewHeight) {
        float fit = Math.max(viewWidth / (float) personBitmap.getWidth(),
                viewHeight / (float) personBitmap.getHeight());
        return personBitmap.getHeight() * fit * personScale;
    }

    private void drawStretched(Canvas canvas, Bitmap bmp, int viewWidth, int viewHeight) {
        float scale = Math.max(viewWidth / (float) bmp.getWidth(),
                viewHeight / (float) bmp.getHeight());
        float frameW = bmp.getWidth() * scale;
        float frameH = bmp.getHeight() * scale;
        canvas.save();
        canvas.translate(viewWidth / 2f, viewHeight / 2f);
        if (mirrored) {
            canvas.scale(-1f, 1f);
        }
        dst.set(-frameW / 2f, -frameH / 2f, frameW / 2f, frameH / 2f);
        canvas.drawBitmap(bmp, null, dst, paint);
        canvas.restore();
    }
}