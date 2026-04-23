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
 * TransUNet Photometric Stereo inferencer.
 *
 * Model input:  NCHW FP32 [1, 3, 512, 512] — 3 grayscale images stacked as channels.
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
     * @return RGB bitmap visualization of the normal map (512x512).
     */
    fun predict(bitmaps: List<Bitmap>): Bitmap {
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

        return tensorToNormalBitmap(output)
    }

    /** Convert 3 bitmaps to tensor [1, 3, 512, 512] with per-image zero-mean unit-std. */
    private fun buildInputTensor(bitmaps: List<Bitmap>): ByteBuffer {
        val pixelCount = INPUT_SIZE * INPUT_SIZE
        val buf = ByteBuffer.allocateDirect(4 * 1 * CHANNELS * pixelCount)
            .order(ByteOrder.nativeOrder())

        bitmaps.forEach { bitmap ->
            val grayscale = bitmapToGrayscaleFloats(bitmap)
            val normalized = zeroMeanUnitStd(grayscale)
            normalized.forEach { buf.putFloat(it) }
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

    /** Convert model output [1, 3, 512, 512] to RGB bitmap via per-pixel L2 norm + [-1,1]→[0,255]. */
    private fun tensorToNormalBitmap(output: ByteBuffer): Bitmap {
        output.rewind()
        val pixelCount = INPUT_SIZE * INPUT_SIZE
        val nx = FloatArray(pixelCount)
        val ny = FloatArray(pixelCount)
        val nz = FloatArray(pixelCount)
        for (i in 0 until pixelCount) nx[i] = output.float
        for (i in 0 until pixelCount) ny[i] = output.float
        for (i in 0 until pixelCount) nz[i] = output.float

        val pixels = IntArray(pixelCount)
        for (i in 0 until pixelCount) {
            val norm = sqrt(nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]) + EPS
            val x = nx[i] / norm
            val y = ny[i] / norm
            val z = nz[i] / norm
            val r = ((x + 1f) * 0.5f * 255f).coerceIn(0f, 255f).toInt()
            val g = ((y + 1f) * 0.5f * 255f).coerceIn(0f, 255f).toInt()
            val b = ((z + 1f) * 0.5f * 255f).coerceIn(0f, 255f).toInt()
            pixels[i] = Color.rgb(r, g, b)
        }
        val bitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        return bitmap
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
