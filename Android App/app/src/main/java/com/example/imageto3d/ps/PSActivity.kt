package com.example.imageto3d.ps

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.imageto3d.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Photometric Stereo capture + inference flow.
 *
 * Flow:
 *   1. User captures 3 grayscale images under 3 different light directions.
 *   2. User taps "Predict" → model runs on-device (~200-300ms on flagship GPU).
 *   3. Result normal map (RGB visualization) is shown over the preview area.
 */
class PSActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var resultView: ImageView
    private lateinit var progress: ProgressBar
    private lateinit var tvInstruction: TextView
    private lateinit var tvStatus: TextView
    private lateinit var btnCapture: Button
    private lateinit var btnPredict: Button
    private lateinit var btnReset: Button
    private lateinit var btnBack: ImageButton
    private val thumbViews = arrayOfNulls<ImageView>(3)

    private var imageCapture: ImageCapture? = null
    private var inferencer: PSInferencer? = null

    private val captured = mutableListOf<Bitmap>()
    private val thumbnails = mutableListOf<Bitmap>()

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, "Cần quyền Camera để chụp ảnh", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout_ps_capture)

        bindViews()
        setupListeners()
        initInferencer()

        if (hasCameraPermission()) startCamera()
        else requestCameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun bindViews() {
        viewFinder = findViewById(R.id.viewFinder)
        resultView = findViewById(R.id.resultView)
        progress = findViewById(R.id.progress)
        tvInstruction = findViewById(R.id.tvInstruction)
        tvStatus = findViewById(R.id.tvStatus)
        btnCapture = findViewById(R.id.btnCapture)
        btnPredict = findViewById(R.id.btnPredict)
        btnReset = findViewById(R.id.btnReset)
        btnBack = findViewById(R.id.btnBack)
        thumbViews[0] = findViewById(R.id.thumb1)
        thumbViews[1] = findViewById(R.id.thumb2)
        thumbViews[2] = findViewById(R.id.thumb3)
    }

    private fun setupListeners() {
        btnCapture.setOnClickListener { takePhoto() }
        btnPredict.setOnClickListener { runInference() }
        btnReset.setOnClickListener { resetState() }
        btnBack.setOnClickListener { finish() }
    }

    private fun initInferencer() {
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { PSInferencer(this@PSActivity) }
            }
            result.onSuccess {
                inferencer = it
                val mode = if (it.isUsingGpu) "GPU (FP16)" else "CPU"
                tvStatus.text = "Model sẵn sàng · $mode"
            }.onFailure {
                Log.e(TAG, "Failed to load model", it)
                tvStatus.text = "Lỗi load model: ${it.message}"
                btnCapture.isEnabled = false
            }
        }
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            runCatching {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            }.onFailure { Log.e(TAG, "Camera bind failed", it) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        if (captured.size >= 3) return

        btnCapture.isEnabled = false
        val executor = ContextCompat.getMainExecutor(this)
        capture.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                image.use { onFrameCaptured(it) }
                btnCapture.isEnabled = true
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e(TAG, "Capture failed", exception)
                Toast.makeText(this@PSActivity, "Chụp lỗi: ${exception.message}", Toast.LENGTH_SHORT).show()
                btnCapture.isEnabled = true
            }
        })
    }

    private fun onFrameCaptured(image: ImageProxy) {
        val raw = ImageUtils.imageProxyToBitmap(image)
        val square = ImageUtils.centerCropSquare(raw, PSInferencer.INPUT_SIZE)
        if (raw !== square) raw.recycle()

        val thumb = ImageUtils.thumbnail(square)
        val index = captured.size
        captured.add(square)
        thumbnails.add(thumb)
        thumbViews[index]?.setImageBitmap(thumb)

        updateInstruction()
    }

    private fun updateInstruction() {
        val count = captured.size
        btnPredict.isEnabled = count == 3 && inferencer != null
        when (count) {
            0 -> tvInstruction.text = "Chụp ảnh 1/3 — đổi hướng đèn, giữ nguyên vật thể"
            1 -> tvInstruction.text = "Chụp ảnh 2/3 — đổi hướng đèn lần nữa"
            2 -> tvInstruction.text = "Chụp ảnh 3/3 — hướng đèn cuối"
            3 -> {
                tvInstruction.text = "Đã đủ 3 ảnh. Nhấn Predict để xử lý."
                btnCapture.isEnabled = false
            }
        }
    }

    private fun runInference() {
        val engine = inferencer ?: run {
            Toast.makeText(this, "Model chưa sẵn sàng", Toast.LENGTH_SHORT).show()
            return
        }
        if (captured.size != 3) return

        setBusy(true)
        val inputs = captured.toList()

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val started = System.nanoTime()
                    val bitmap = engine.predict(inputs)
                    val ms = (System.nanoTime() - started) / 1_000_000
                    bitmap to ms
                }
            }
            setBusy(false)
            result.onSuccess { (bitmap, ms) ->
                resultView.setImageBitmap(bitmap)
                resultView.visibility = View.VISIBLE
                tvInstruction.text = "Kết quả normal map"
                tvStatus.text = "Inference: ${ms}ms · ${if (engine.isUsingGpu) "GPU" else "CPU"}"
            }.onFailure {
                Log.e(TAG, "Inference failed", it)
                Toast.makeText(this@PSActivity, "Lỗi inference: ${it.message}", Toast.LENGTH_LONG).show()
                tvStatus.text = "Inference thất bại"
            }
        }
    }

    private fun setBusy(busy: Boolean) {
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        btnCapture.isEnabled = !busy && captured.size < 3
        btnPredict.isEnabled = !busy && captured.size == 3
        btnReset.isEnabled = !busy
    }

    private fun resetState() {
        captured.forEach { if (!it.isRecycled) it.recycle() }
        thumbnails.forEach { if (!it.isRecycled) it.recycle() }
        captured.clear()
        thumbnails.clear()
        thumbViews.forEach { it?.setImageBitmap(null) }
        resultView.visibility = View.GONE
        resultView.setImageBitmap(null)
        btnCapture.isEnabled = true
        updateInstruction()
        tvStatus.text = ""
    }

    override fun onDestroy() {
        super.onDestroy()
        inferencer?.close()
        captured.forEach { if (!it.isRecycled) it.recycle() }
        thumbnails.forEach { if (!it.isRecycled) it.recycle() }
    }

    companion object {
        private const val TAG = "PSActivity"
    }
}
