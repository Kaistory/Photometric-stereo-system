package com.example.imageto3d.ps

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal writer for NumPy .npy format (version 1.0, FP32 little-endian, C-order).
 *
 * Format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 */
object NumpyWriter {

    private const val MAGIC = "NUMPY"
    private const val VERSION_MAJOR: Byte = 1
    private const val VERSION_MINOR: Byte = 0
    private const val HEADER_ALIGNMENT = 64

    /**
     * Save a float32 array to [destination] in NumPy .npy format with the given [shape].
     * [data] is in C-order (row-major) matching the given shape.
     */
    fun saveFloat32(destination: File, data: FloatArray, shape: IntArray) {
        val expectedSize = shape.fold(1, Int::times)
        require(data.size == expectedSize) {
            "Data size ${data.size} does not match shape product $expectedSize"
        }

        val shapeStr = shape.joinToString(", ") { "$it" }.let { if (shape.size == 1) "$it," else it }
        val headerDict = "{'descr': '<f4', 'fortran_order': False, 'shape': ($shapeStr), }"

        val preludeSize = MAGIC.length + 2 + 2
        val dictBytes = headerDict.toByteArray(Charsets.US_ASCII)
        val totalWithoutPadding = preludeSize + dictBytes.size + 1
        val padding = (HEADER_ALIGNMENT - totalWithoutPadding % HEADER_ALIGNMENT) % HEADER_ALIGNMENT
        val paddedHeader = ByteArray(dictBytes.size + padding + 1).apply {
            dictBytes.copyInto(this, 0)
            for (i in dictBytes.size until size - 1) this[i] = ' '.code.toByte()
            this[size - 1] = '\n'.code.toByte()
        }

        require(paddedHeader.size < 0x10000) { "Header too large for npy v1.0" }

        FileOutputStream(destination).use { out ->
            out.write(MAGIC.toByteArray(Charsets.ISO_8859_1))
            out.write(byteArrayOf(VERSION_MAJOR, VERSION_MINOR))
            out.write(paddedHeader.size and 0xFF)
            out.write((paddedHeader.size shr 8) and 0xFF)
            out.write(paddedHeader)

            val byteBuffer = ByteBuffer.allocate(4 * data.size).order(ByteOrder.LITTLE_ENDIAN)
            for (v in data) byteBuffer.putFloat(v)
            out.write(byteBuffer.array())
        }
    }
}
