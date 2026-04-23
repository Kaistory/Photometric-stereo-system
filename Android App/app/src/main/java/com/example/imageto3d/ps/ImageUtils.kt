package com.example.imageto3d.ps

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.YuvImage

object ImageUtils {

    /**
     * Convert a CameraX ImageProxy to a Bitmap, applying rotation so the result is upright.
     */
    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val bitmap = when (image.format) {
            ImageFormat.YUV_420_888 -> yuv420ToBitmap(image)
            else -> jpegProxyToBitmap(image)
        }
        val rotation = image.imageInfo.rotationDegrees
        return if (rotation != 0) rotate(bitmap, rotation.toFloat()) else bitmap
    }

    private fun jpegProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun yuv420ToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 95, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun rotate(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Center-crop the bitmap to a square, then scale to [targetSize] x [targetSize].
     */
    fun centerCropSquare(source: Bitmap, targetSize: Int): Bitmap {
        val shorter = minOf(source.width, source.height)
        val xOffset = (source.width - shorter) / 2
        val yOffset = (source.height - shorter) / 2

        val square = Bitmap.createBitmap(source, xOffset, yOffset, shorter, shorter)
        val scaled = Bitmap.createScaledBitmap(square, targetSize, targetSize, true)
        if (square !== source && square !== scaled) square.recycle()
        return scaled
    }

    /**
     * Build a 128x128 thumbnail for on-screen preview of captured shots.
     */
    fun thumbnail(source: Bitmap, size: Int = 128): Bitmap {
        val thumb = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(thumb)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        canvas.drawBitmap(source, null, Rect(0, 0, size, size), paint)
        return thumb
    }
}
