package com.example.imageto3d.ps

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Raw photometric stereo inference result.
 *
 * @property visualization RGB bitmap of the normal map for on-screen display (512x512 ARGB).
 * @property rawNormal Flat float32 array in HWC layout: for pixel (h, w) the xyz values are
 *   rawNormal[(h * width + w) * 3 + {0,1,2}]. Values are already L2-normalized, range roughly [-1, 1].
 * @property width Width of the result (same as height for this model).
 * @property height Height of the result.
 */
data class PSResult(
    val visualization: Bitmap,
    val rawNormal: FloatArray,
    val width: Int,
    val height: Int,
)

/**
 * TransUNet Photometric Stereo inferencer (INT8-quantized model, 14MB).
 *
 * Model input:  NHWC FP32 [1, 512, 512, 3] — 3 grayscale images interleaved per pixel.
 *               Each image pre-normalized to zero-mean, unit-std.
 * Model output: NCHW FP32 [1, 3, 512, 512] — normal map (nx, ny, nz) in roughly [-1, 1].
 */
class PSInferencer(context: Context) : AutoCloseable {

    companion object {
        const val INPUT_SIZE = 512
        const val CHANNELS = 3
        private const val MODEL_FILE = "model.tflite"
        private const val EPS = 1e-8f
        private const val TAG = "PSInferencer"
    }

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?
    val isUsingGpu: Boolean get() = gpuDelegate != null

    init {
        val modelBuffer = loadModelFile(context)
        val (interp, delegate) = buildInterpreter(modelBuffer)
        interpreter = interp
        gpuDelegate = delegate
        logIO()
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        FileInputStream(fd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    private fun buildInterpreter(model: MappedByteBuffer): Pair<Interpreter, GpuDelegate?> {
        val gpuSupported = runCatching { CompatibilityList().isDelegateSupportedOnThisDevice }.getOrDefault(false)
        if (gpuSupported) {
            val gpuAttempt = runCatching {
                // Use default GpuDelegate() — FP16 precision loss flag is not exposed
                // in litert-gpu:1.0.1 (its GpuDelegateFactory.Options parent class is
                // not shipped). GPU still gives the bulk of the speedup vs CPU.
                val delegate = GpuDelegate()
                val opts = Interpreter.Options().apply {
                    addDelegate(delegate)
                    numThreads = 4
                }
                Interpreter(model, opts) to delegate
            }
            if (gpuAttempt.isSuccess) {
                Log.i(TAG, "Using GPU delegate (FP16)")
                return gpuAttempt.getOrThrow()
            }
            Log.w(TAG, "GPU delegate init failed, falling back to CPU: ${gpuAttempt.exceptionOrNull()?.message}")
        }
        val cpuOpts = Interpreter.Options().apply {
            numThreads = 4
        }
        Log.i(TAG, "Using CPU interpreter (XNNPACK, 4 threads)")
        return Interpreter(model, cpuOpts) to null
    }

    private fun logIO() {
        val inputShape = interpreter.getInputTensor(0).shape().toList()
        val outputShape = interpreter.getOutputTensor(0).shape().toList()
        Log.i(TAG, "Model input: shape=$inputShape, output: shape=$outputShape")
    }

    /**
     * Predict normal map from 3 grayscale bitmaps (each exactly 512x512).
     *
     * @return [PSResult] with both RGB visualization bitmap and raw float32 normal map (HWC).
     */
    fun predict(bitmaps: List<Bitmap>): PSResult {
        require(bitmaps.size == CHANNELS) { "Need exactly $CHANNELS images, got ${bitmaps.size}" }
        require(bitmaps.all { it.width == INPUT_SIZE && it.height == INPUT_SIZE }) {
            "All images must be ${INPUT_SIZE}x$INPUT_SIZE"
        }

        val input = buildInputTensor(bitmaps)
        val output = ByteBuffer.allocateDirect(4 * 1 * CHANNELS * INPUT_SIZE * INPUT_SIZE)
            .order(ByteOrder.nativeOrder())

        val start = System.nanoTime()
        interpreter.run(input, output)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        Log.i(TAG, "Inference took ${elapsedMs}ms")

        return buildResult(output)
    }

    /**
     * Convert 3 bitmaps to tensor [1, 512, 512, 3] NHWC with per-image zero-mean unit-std.
     * Layout: for each pixel index i, emit (img1[i], img2[i], img3[i]) interleaved.
     */
    private fun buildInputTensor(bitmaps: List<Bitmap>): ByteBuffer {
        val pixelCount = INPUT_SIZE * INPUT_SIZE
        val buf = ByteBuffer.allocateDirect(4 * 1 * pixelCount * CHANNELS)
            .order(ByteOrder.nativeOrder())

        val c0 = zeroMeanUnitStd(bitmapToGrayscaleFloats(bitmaps[0]))
        val c1 = zeroMeanUnitStd(bitmapToGrayscaleFloats(bitmaps[1]))
        val c2 = zeroMeanUnitStd(bitmapToGrayscaleFloats(bitmaps[2]))

        for (i in 0 until pixelCount) {
            buf.putFloat(c0[i])
            buf.putFloat(c1[i])
            buf.putFloat(c2[i])
        }
        buf.rewind()
        return buf
    }

    private fun bitmapToGrayscaleFloats(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val out = FloatArray(pixels.size)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            out[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f
        }
        return out
    }

    private fun zeroMeanUnitStd(values: FloatArray): FloatArray {
        var sum = 0.0
        for (v in values) sum += v
        val mean = (sum / values.size).toFloat()

        var sqSum = 0.0
        for (v in values) {
            val d = v - mean
            sqSum += d * d
        }
        val std = sqrt(sqSum / values.size).toFloat() + EPS

        val out = FloatArray(values.size)
        for (i in values.indices) out[i] = (values[i] - mean) / std
        return out
    }

    /**
     * Convert model output [1, 3, 512, 512] NCHW to:
     *   - Normalized HWC float array (ready for .npy export, matches Python reference layout)
     *   - RGB Bitmap for on-screen display
     *
     * Applies per-pixel L2 normalization and maps each unit vector component [-1,1] → [0,255].
     */
    private fun buildResult(output: ByteBuffer): PSResult {
        output.rewind()
        val pixelCount = INPUT_SIZE * INPUT_SIZE
        val nx = FloatArray(pixelCount)
        val ny = FloatArray(pixelCount)
        val nz = FloatArray(pixelCount)
        for (i in 0 until pixelCount) nx[i] = output.float
        for (i in 0 until pixelCount) ny[i] = output.float
        for (i in 0 until pixelCount) nz[i] = output.float

        val hwc = FloatArray(pixelCount * CHANNELS)
        val pixels = IntArray(pixelCount)
        for (i in 0 until pixelCount) {
            val norm = sqrt(nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]) + EPS
            val x = nx[i] / norm
            val y = ny[i] / norm
            val z = nz[i] / norm
            val base = i * CHANNELS
            hwc[base] = x
            hwc[base + 1] = y
            hwc[base + 2] = z
            val r = ((x + 1f) * 0.5f * 255f).coerceIn(0f, 255f).toInt()
            val g = ((y + 1f) * 0.5f * 255f).coerceIn(0f, 255f).toInt()
            val b = ((z + 1f) * 0.5f * 255f).coerceIn(0f, 255f).toInt()
            pixels[i] = Color.rgb(r, g, b)
        }
        val bitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        return PSResult(visualization = bitmap, rawNormal = hwc, width = INPUT_SIZE, height = INPUT_SIZE)
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
