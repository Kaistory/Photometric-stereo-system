package com.example.imageto3d

import android.Manifest
import android.app.AlertDialog
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
class MainActivity : AppCompatActivity() {

    private lateinit var btnConnectBle: Button
    private lateinit var tvBleStatus: TextView
    private val PERMISSION_REQUEST_CODE = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.layout_led_control)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.ledControlLayout)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        btnConnectBle = findViewById(R.id.btnConnectBle)
        tvBleStatus = findViewById(R.id.tvBleStatus)

        // 2. Bắt sự kiện Click
        btnConnectBle.setOnClickListener {
            if (hasBluetoothPermissions()) {
                // Nếu đã có quyền -> Hiển thị Pop-up
                showBluetoothDeviceListDialog()
            } else {
                // Nếu chưa có quyền -> Yêu cầu cấp quyền
                requestBluetoothPermissions()
            }
        }

    }

    private fun showBluetoothDeviceListDialog() {
        // TODO: Ở bước sau, bạn sẽ thay mảng này bằng danh sách thiết bị quét được từ BLE Scanner
        val deviceNames = arrayOf(
            "ESP32-C3_LED_Control",
            "ESP32_Test_Board",
            "Unknown Device"
        )

        // Tạo Pop-up
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Chọn thiết bị ESP32 để kết nối")

        // Nạp danh sách vào Pop-up và bắt sự kiện khi user chọn 1 dòng
        builder.setItems(deviceNames) { dialog, which ->
            val selectedDeviceName = deviceNames[which]

            // Cập nhật giao diện
            tvBleStatus.text = "Đang kết nối: $selectedDeviceName..."
            Toast.makeText(this, "Đã chọn: $selectedDeviceName", Toast.LENGTH_SHORT).show()

            // TODO: Gọi hàm thực hiện kết nối GATT (BLE) tới địa chỉ MAC của thiết bị này
        }

        // Thêm nút Hủy
        builder.setNegativeButton("Hủy") { dialog, _ ->
            dialog.dismiss()
        }

        // Hiển thị lên màn hình
        val dialog = builder.create()
        dialog.show()
    }

    // --- CÁC HÀM XỬ LÝ QUYỀN (RẤT QUAN TRỌNG) ---
    private fun hasBluetoothPermissions(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestBluetoothPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN),
                PERMISSION_REQUEST_CODE
            )
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                PERMISSION_REQUEST_CODE
            )
        }
    }
}