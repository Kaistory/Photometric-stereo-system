package com.example.ledcontrol

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.ledcontrol.ui.theme.LedControlTheme
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.eclipse.paho.client.mqttv3.*
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence
import java.util.UUID
import kotlin.math.cos
import kotlin.math.sin

// MQTT Configuration
const val MQTT_SERVER = "ssl://76bf78e5731240cbb090e3284089c085.s1.eu.hivemq.cloud:8883"
const val MQTT_USER = "hivemq.webclient.1765688353696"
const val MQTT_PASS = "r8>!lHMZ37RYzy?5<Sxv"
const val TOPIC_PREFIX = "esp32/controlled/"

// Beautiful color palette
val DarkBackground = Color(0xFF0D1117)
val CardBackground = Color(0xFF161B22)
val AccentCyan = Color(0xFF00D9FF)
val AccentPurple = Color(0xFF8B5CF6)
val AccentPink = Color(0xFFEC4899)
val AccentGreen = Color(0xFF10B981)

// Data class for LED state
data class LedState(
    val red: Int,
    val green: Int,
    val blue: Int,
    val brightness: Int
)

// Data class for color palette
data class ColorPalette(
    val name: String,
    val icon: String,
    val colors: List<LedState>,
    val isCustom: Boolean = false
)

// Data class for serializable custom preset
data class CustomPreset(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val icon: String,
    val ledStates: List<LedState>,
    val createdAt: Long = System.currentTimeMillis()
)

// Helper class to manage custom presets with SharedPreferences
class PresetManager(context: Context) {
    private val sharedPrefs: SharedPreferences =
        context.getSharedPreferences("led_presets", Context.MODE_PRIVATE)
    private val gson = Gson()
    private val presetsKey = "custom_presets"

    fun savePreset(preset: CustomPreset) {
        val presets = getPresets().toMutableList()
        presets.add(preset)
        savePresetsList(presets)
    }

    fun deletePreset(presetId: String) {
        val presets = getPresets().toMutableList()
        presets.removeAll { it.id == presetId }
        savePresetsList(presets)
    }

    fun getPresets(): List<CustomPreset> {
        val json = sharedPrefs.getString(presetsKey, null) ?: return emptyList()
        return try {
            val type = object : TypeToken<List<CustomPreset>>() {}.type
            gson.fromJson(json, type) ?: emptyList()
        } catch (e: Exception) {
            Log.e("PresetManager", "Error loading presets: ${e.message}")
            emptyList()
        }
    }

    private fun savePresetsList(presets: List<CustomPreset>) {
        val json = gson.toJson(presets)
        sharedPrefs.edit().putString(presetsKey, json).apply()
    }
}

// Helper function to convert HSV to RGB
fun hsvToRgb(h: Float, s: Float, v: Float): Triple<Int, Int, Int> {
    val c = v * s
    val x = c * (1 - kotlin.math.abs((h / 60) % 2 - 1))
    val m = v - c

    val (r, g, b) = when {
        h < 60 -> Triple(c, x, 0f)
        h < 120 -> Triple(x, c, 0f)
        h < 180 -> Triple(0f, c, x)
        h < 240 -> Triple(0f, x, c)
        h < 300 -> Triple(x, 0f, c)
        else -> Triple(c, 0f, x)
    }

    return Triple(
        ((r + m) * 255).toInt().coerceIn(0, 255),
        ((g + m) * 255).toInt().coerceIn(0, 255),
        ((b + m) * 255).toInt().coerceIn(0, 255)
    )
}

// Pre-set color palettes
val presetPalettes = listOf(
    ColorPalette(
        name = "Rainbow",
        icon = "🌈",
        colors = List(24) { index ->
            val hue = (index * 15f) % 360f
            val rgb = hsvToRgb(hue, 1f, 1f)
            LedState(rgb.first, rgb.second, rgb.third, 200)
        }
    ),
    ColorPalette(
        name = "Ocean Wave",
        icon = "🌊",
        colors = List(24) { index ->
            val phase = (index * 15f) % 360f
            val blue = (200 + 55 * sin(Math.toRadians(phase.toDouble()))).toInt()
            val green = (100 + 100 * sin(Math.toRadians((phase + 60).toDouble()))).toInt()
            LedState(0, green.coerceIn(0, 255), blue.coerceIn(0, 255), 180)
        }
    ),
    ColorPalette(
        name = "Fire",
        icon = "🔥",
        colors = List(24) { index ->
            val phase = (index * 15f) % 360f
            val red = 255
            val green = (80 + 80 * sin(Math.toRadians(phase.toDouble()))).toInt()
            LedState(red, green.coerceIn(0, 160), 0, 220)
        }
    ),
    ColorPalette(
        name = "Aurora",
        icon = "✨",
        colors = List(24) { index ->
            val phase = index * 15f
            val green = (150 + 105 * sin(Math.toRadians(phase.toDouble()))).toInt()
            val blue = (100 + 155 * cos(Math.toRadians(phase.toDouble()))).toInt()
            val red = (50 + 50 * sin(Math.toRadians((phase * 2).toDouble()))).toInt()
            LedState(red.coerceIn(0, 255), green.coerceIn(0, 255), blue.coerceIn(0, 255), 200)
        }
    ),
    ColorPalette(
        name = "Sunset",
        icon = "🌅",
        colors = List(24) { index ->
            val phase = index * 15f
            val red = 255
            val green = (100 + 80 * sin(Math.toRadians(phase.toDouble()))).toInt()
            val blue = (50 + 100 * cos(Math.toRadians((phase + 90).toDouble()))).toInt()
            LedState(red, green.coerceIn(50, 180), blue.coerceIn(0, 150), 200)
        }
    ),
    ColorPalette(
        name = "Christmas",
        icon = "🎄",
        colors = List(24) { index ->
            if (index % 2 == 0) LedState(255, 0, 0, 200) else LedState(0, 255, 0, 200)
        }
    ),
    ColorPalette(
        name = "Party",
        icon = "🎉",
        colors = List(24) { index ->
            when (index % 4) {
                0 -> LedState(255, 0, 255, 220) // Magenta
                1 -> LedState(0, 255, 255, 220) // Cyan
                2 -> LedState(255, 255, 0, 220) // Yellow
                else -> LedState(255, 100, 0, 220) // Orange
            }
        }
    ),
    ColorPalette(
        name = "Cool White",
        icon = "❄️",
        colors = List(24) { LedState(255, 255, 255, 180) }
    ),
    ColorPalette(
        name = "Warm White",
        icon = "💡",
        colors = List(24) { LedState(255, 200, 150, 180) }
    ),
    ColorPalette(
        name = "All Off",
        icon = "⚫",
        colors = List(24) { LedState(0, 0, 0, 0) }
    )
)

class MainActivity : ComponentActivity() {
    private lateinit var mqttClient: MqttClient

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        connectMqtt()

        setContent {
            LedControlTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = DarkBackground
                ) {
                    LedControlScreen(
                        onSendColor = { ledId, r, g, b, brightness ->
                            publishColor(ledId, r, g, b, brightness)
                        },
                        onSendAllColors = { states ->
                            // Send all LED colors with a small delay between each
                            Thread {
                                states.forEachIndexed { index, state ->
                                    publishColor(index.toString(), state.red, state.green, state.blue, state.brightness)
                                    Thread.sleep(50) // Small delay to prevent overwhelming MQTT
                                }
                            }.start()
                        }
                    )
                }
            }
        }
    }

    private fun connectMqtt() {
        Thread {
            try {
                val clientId = "AndroidClient_${UUID.randomUUID()}"
                mqttClient = MqttClient(MQTT_SERVER, clientId, MemoryPersistence())

                val options = MqttConnectOptions().apply {
                    userName = MQTT_USER
                    password = MQTT_PASS.toCharArray()
                    isCleanSession = true
                    connectionTimeout = 30
                    keepAliveInterval = 60
                }

                mqttClient.connect(options)
                Log.d("MQTT", "Connected successfully!")
            } catch (e: Exception) {
                Log.e("MQTT", "Connection failed: ${e.message}")
            }
        }.start()
    }

    private fun publishColor(ledId: String, r: Int, g: Int, b: Int, brightness: Int) {
        Thread {
            try {
                if (::mqttClient.isInitialized && mqttClient.isConnected) {
                    val topic = "$TOPIC_PREFIX$ledId"
                    val payload = "$r,$g,$b,$brightness"
                    mqttClient.publish(topic, payload.toByteArray(), 1, false)
                    Log.d("MQTT", "Published to $topic: $payload")
                }
            } catch (e: Exception) {
                Log.e("MQTT", "Publish failed: ${e.message}")
            }
        }.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            if (::mqttClient.isInitialized && mqttClient.isConnected) {
                mqttClient.disconnect()
            }
        } catch (e: Exception) {
            Log.e("MQTT", "Disconnect failed: ${e.message}")
        }
    }
}

@Composable
fun LedControlScreen(
    modifier: Modifier = Modifier,
    onSendColor: (String, Int, Int, Int, Int) -> Unit,
    onSendAllColors: (List<LedState>) -> Unit
) {
    val context = LocalContext.current
    val presetManager = remember { PresetManager(context) }

    var selectedLed by remember { mutableStateOf(0) }
    var ledStates by remember { mutableStateOf(List(24) { LedState(255, 100, 50, 200) }) }
    var red by remember { mutableStateOf(255f) }
    var green by remember { mutableStateOf(100f) }
    var blue by remember { mutableStateOf(50f) }
    var brightness by remember { mutableStateOf(200f) }
    var showPaletteSelector by remember { mutableStateOf(false) }

    // Custom preset states
    var customPresets by remember { mutableStateOf(presetManager.getPresets()) }
    var showSaveDialog by remember { mutableStateOf(false) }
    var showDeleteConfirmDialog by remember { mutableStateOf<CustomPreset?>(null) }

    // Global brightness mode - when true, brightness slider affects all LEDs
    var globalBrightnessMode by remember { mutableStateOf(false) }
    var globalBrightness by remember { mutableStateOf(200f) }

    // Preview color with animation
    val previewColor = Color(
        red = red.toInt(),
        green = green.toInt(),
        blue = blue.toInt(),
        alpha = 255
    )

    // Animated glow effect
    val infiniteTransition = rememberInfiniteTransition(label = "glow")
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.5f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glowAlpha"
    )

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
            .padding(top = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Title with gradient
        Text(
            text = "✨ LED Controller",
            style = MaterialTheme.typography.headlineMedium.copy(
                fontWeight = FontWeight.Bold,
                brush = Brush.linearGradient(
                    colors = listOf(AccentCyan, AccentPurple, AccentPink)
                )
            )
        )

        // Circular LED Ring with glow effect
        Box(
            modifier = Modifier.size(280.dp),
            contentAlignment = Alignment.Center
        ) {
            // Background glow
            Canvas(modifier = Modifier.size(280.dp)) {
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            previewColor.copy(alpha = glowAlpha * 0.5f),
                            Color.Transparent
                        )
                    ),
                    radius = size.minDimension / 2
                )
            }

            // Decorative ring
            Canvas(modifier = Modifier.size(260.dp)) {
                drawCircle(
                    brush = Brush.sweepGradient(
                        colors = listOf(
                            AccentCyan.copy(alpha = 0.3f),
                            AccentPurple.copy(alpha = 0.3f),
                            AccentPink.copy(alpha = 0.3f),
                            AccentCyan.copy(alpha = 0.3f)
                        )
                    ),
                    style = Stroke(width = 2.dp.toPx())
                )
            }

            CircularLedLayout(
                ledCount = 24,
                selectedLed = selectedLed,
                ledStates = ledStates,
                onLedClick = { index ->
                    ledStates = ledStates.toMutableList().apply {
                        this[selectedLed] = LedState(red.toInt(), green.toInt(), blue.toInt(), brightness.toInt())
                    }
                    selectedLed = index
                    val newState = ledStates[index]
                    red = newState.red.toFloat()
                    green = newState.green.toFloat()
                    blue = newState.blue.toFloat()
                    brightness = newState.brightness.toFloat()
                }
            )

            // Center preview with glow
            Box(contentAlignment = Alignment.Center) {
                // Glow behind preview
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .blur(20.dp)
                        .background(previewColor.copy(alpha = glowAlpha), CircleShape)
                )
                // Main preview circle
                Box(
                    modifier = Modifier
                        .size(90.dp)
                        .shadow(8.dp, CircleShape)
                        .clip(CircleShape)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(
                                    previewColor,
                                    previewColor.copy(alpha = 0.8f)
                                )
                            )
                        )
                        .border(3.dp, Color.White.copy(alpha = 0.3f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            imageVector = Icons.Filled.Lightbulb,
                            contentDescription = "LED",
                            tint = if ((red + green + blue) / 3 > 128) Color.Black.copy(alpha = 0.7f) else Color.White,
                            modifier = Modifier.size(32.dp)
                        )
                        Text(
                            text = "LED $selectedLed",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = if ((red + green + blue) / 3 > 128) Color.Black.copy(alpha = 0.8f) else Color.White
                        )
                    }
                }
            }
        }

        // Info Card with glassmorphism effect
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .shadow(8.dp, RoundedCornerShape(16.dp)),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = CardBackground)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                InfoChip(label = "R", value = red.toInt(), color = Color.Red)
                InfoChip(label = "G", value = green.toInt(), color = AccentGreen)
                InfoChip(label = "B", value = blue.toInt(), color = Color(0xFF3B82F6))
                InfoChip(label = "☀", value = brightness.toInt(), color = Color.Yellow)
            }
        }

        // Sliders with beautiful design
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(CardBackground, RoundedCornerShape(16.dp))
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            GradientSlider(
                label = "Red",
                value = red,
                onValueChange = { red = it },
                gradientColors = listOf(Color(0xFF1a0000), Color.Red)
            )
            GradientSlider(
                label = "Green",
                value = green,
                onValueChange = { green = it },
                gradientColors = listOf(Color(0xFF001a00), AccentGreen)
            )
            GradientSlider(
                label = "Blue",
                value = blue,
                onValueChange = { blue = it },
                gradientColors = listOf(Color(0xFF00001a), Color(0xFF3B82F6))
            )

            HorizontalDivider(
                modifier = Modifier.padding(vertical = 8.dp),
                color = Color.White.copy(alpha = 0.1f)
            )

            // Global Brightness Mode Toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "☀️ ",
                        fontSize = 16.sp
                    )
                    Text(
                        text = if (globalBrightnessMode) "All LEDs Brightness" else "LED $selectedLed Brightness",
                        color = Color.White,
                        fontWeight = FontWeight.Medium,
                        fontSize = 14.sp
                    )
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "All",
                        color = if (globalBrightnessMode) AccentCyan else Color.White.copy(alpha = 0.5f),
                        fontSize = 12.sp
                    )
                    Switch(
                        checked = globalBrightnessMode,
                        onCheckedChange = {
                            globalBrightnessMode = it
                            if (it) {
                                // When switching to global mode, use current brightness as global
                                globalBrightness = brightness
                            }
                        },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = AccentCyan,
                            checkedTrackColor = AccentCyan.copy(alpha = 0.5f),
                            uncheckedThumbColor = Color.Gray,
                            uncheckedTrackColor = Color.Gray.copy(alpha = 0.3f)
                        ),
                        modifier = Modifier.padding(horizontal = 8.dp)
                    )
                }
            }

            // Brightness Slider - behavior depends on globalBrightnessMode
            if (globalBrightnessMode) {
                GradientSlider(
                    label = "Global Brightness",
                    value = globalBrightness,
                    onValueChange = { newBrightness ->
                        globalBrightness = newBrightness
                        // Update all LEDs brightness in real-time
                        ledStates = ledStates.map { state ->
                            state.copy(brightness = newBrightness.toInt())
                        }
                        brightness = newBrightness
                    },
                    gradientColors = listOf(Color(0xFF1a1a1a), Color.Yellow)
                )

                // Apply to All button for global brightness
                Button(
                    onClick = {
                        // Send all LED colors with the new global brightness
                        onSendAllColors(ledStates)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Yellow.copy(alpha = 0.2f)
                    ),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = "☀️ Apply Brightness to All LEDs",
                        color = Color.Yellow,
                        fontWeight = FontWeight.Medium
                    )
                }
            } else {
                GradientSlider(
                    label = "Brightness",
                    value = brightness,
                    onValueChange = { brightness = it },
                    gradientColors = listOf(Color(0xFF1a1a1a), Color.Yellow)
                )
            }
        }

        // Preset Palette Selector
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .shadow(4.dp, RoundedCornerShape(12.dp)),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = CardBackground)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "🎨 Color Presets",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                    Row {
                        // Save current state button
                        IconButton(
                            onClick = { showSaveDialog = true }
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Save,
                                contentDescription = "Save Preset",
                                tint = AccentGreen
                            )
                        }
                        TextButton(
                            onClick = { showPaletteSelector = !showPaletteSelector }
                        ) {
                            Text(
                                text = if (showPaletteSelector) "Hide ▲" else "Show ▼",
                                color = AccentCyan,
                                fontSize = 12.sp
                            )
                        }
                    }
                }

                if (showPaletteSelector) {
                    // Custom Presets Section
                    if (customPresets.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "💾 My Presets",
                            color = Color.White.copy(alpha = 0.7f),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        LazyRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(customPresets, key = { it.id }) { preset ->
                                CustomPresetItem(
                                    preset = preset,
                                    onClick = {
                                        ledStates = preset.ledStates
                                        val newState = ledStates[selectedLed]
                                        red = newState.red.toFloat()
                                        green = newState.green.toFloat()
                                        blue = newState.blue.toFloat()
                                        brightness = newState.brightness.toFloat()
                                        onSendAllColors(ledStates)
                                    },
                                    onLongClick = {
                                        showDeleteConfirmDialog = preset
                                    }
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        HorizontalDivider(color = Color.White.copy(alpha = 0.1f))
                    }

                    // Built-in Presets Section
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "🎨 Built-in Presets",
                        color = Color.White.copy(alpha = 0.7f),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(presetPalettes) { palette ->
                            PresetPaletteItem(
                                palette = palette,
                                onClick = {
                                    ledStates = palette.colors
                                    // Update current sliders to match selected LED
                                    val newState = ledStates[selectedLed]
                                    red = newState.red.toFloat()
                                    green = newState.green.toFloat()
                                    blue = newState.blue.toFloat()
                                    brightness = newState.brightness.toFloat()
                                    // Send all colors to ESP32
                                    onSendAllColors(ledStates)
                                }
                            )
                        }
                    }
                }
            }
        }

        // Send Button with gradient
        Button(
            onClick = {
                ledStates = ledStates.toMutableList().apply {
                    this[selectedLed] = LedState(red.toInt(), green.toInt(), blue.toInt(), brightness.toInt())
                }
                onSendColor(selectedLed.toString(), red.toInt(), green.toInt(), blue.toInt(), brightness.toInt())
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        Brush.linearGradient(
                            colors = listOf(AccentCyan, AccentPurple, AccentPink)
                        ),
                        RoundedCornerShape(16.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "SEND TO LED $selectedLed",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))
    }

    // Save Preset Dialog
    if (showSaveDialog) {
        SavePresetDialog(
            onDismiss = { showSaveDialog = false },
            onSave = { name, icon ->
                // Update current LED state before saving
                val updatedStates = ledStates.toMutableList().apply {
                    this[selectedLed] = LedState(red.toInt(), green.toInt(), blue.toInt(), brightness.toInt())
                }
                ledStates = updatedStates

                val preset = CustomPreset(
                    name = name,
                    icon = icon,
                    ledStates = updatedStates
                )
                presetManager.savePreset(preset)
                customPresets = presetManager.getPresets()
                showSaveDialog = false
            }
        )
    }

    // Delete Confirmation Dialog
    showDeleteConfirmDialog?.let { preset ->
        DeletePresetDialog(
            presetName = preset.name,
            onDismiss = { showDeleteConfirmDialog = null },
            onConfirm = {
                presetManager.deletePreset(preset.id)
                customPresets = presetManager.getPresets()
                showDeleteConfirmDialog = null
            }
        )
    }
}

@Composable
fun PresetPaletteItem(
    palette: ColorPalette,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .width(80.dp)
            .clickable { onClick() },
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF21262D))
    ) {
        Column(
            modifier = Modifier.padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = palette.icon,
                fontSize = 24.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = palette.name,
                color = Color.White,
                fontSize = 10.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1
            )
            Spacer(modifier = Modifier.height(4.dp))
            // Preview of palette colors
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                repeat(4) { i ->
                    val state = palette.colors[i * 6]
                    Box(
                        modifier = Modifier
                            .size(12.dp)
                            .background(
                                Color(state.red, state.green, state.blue),
                                CircleShape
                            )
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun CustomPresetItem(
    preset: CustomPreset,
    onClick: () -> Unit,
    onLongClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .width(80.dp)
            .combinedClickable(
                onClick = onClick,
                onLongClick = onLongClick
            ),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF21262D))
    ) {
        Column(
            modifier = Modifier.padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = preset.icon,
                fontSize = 24.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = preset.name,
                color = Color.White,
                fontSize = 10.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1
            )
            Spacer(modifier = Modifier.height(4.dp))
            // Preview of preset colors
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                repeat(4) { i ->
                    val state = preset.ledStates[i * 6]
                    Box(
                        modifier = Modifier
                            .size(12.dp)
                            .background(
                                Color(state.red, state.green, state.blue),
                                CircleShape
                            )
                    )
                }
            }
        }
    }
}

@Composable
fun SavePresetDialog(
    onDismiss: () -> Unit,
    onSave: (name: String, icon: String) -> Unit
) {
    var presetName by remember { mutableStateOf("") }
    var selectedIcon by remember { mutableStateOf("💾") }

    val iconOptions = listOf("💾", "⭐", "❤️", "💜", "💙", "💚", "🧡", "🌟", "🎵", "🏠", "🎮", "🌙")

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = CardBackground)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Save Current State",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Icon selector
                Text(
                    text = "Choose an icon",
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 12.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(iconOptions) { icon ->
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .background(
                                    if (icon == selectedIcon) AccentPurple.copy(alpha = 0.3f)
                                    else Color.Transparent,
                                    CircleShape
                                )
                                .border(
                                    width = if (icon == selectedIcon) 2.dp else 0.dp,
                                    color = if (icon == selectedIcon) AccentPurple else Color.Transparent,
                                    shape = CircleShape
                                )
                                .clickable { selectedIcon = icon },
                            contentAlignment = Alignment.Center
                        ) {
                            Text(text = icon, fontSize = 20.sp)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Name input
                OutlinedTextField(
                    value = presetName,
                    onValueChange = { presetName = it },
                    label = { Text("Preset Name", color = Color.White.copy(alpha = 0.7f)) },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = AccentCyan,
                        unfocusedBorderColor = Color.White.copy(alpha = 0.3f),
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        cursorColor = AccentCyan
                    ),
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(20.dp))

                // Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color.White
                        )
                    ) {
                        Text("Cancel")
                    }
                    Button(
                        onClick = {
                            if (presetName.isNotBlank()) {
                                onSave(presetName.trim(), selectedIcon)
                            }
                        },
                        modifier = Modifier.weight(1f),
                        enabled = presetName.isNotBlank(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = AccentGreen,
                            contentColor = Color.White
                        )
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Save,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Save")
                    }
                }
            }
        }
    }
}

@Composable
fun DeletePresetDialog(
    presetName: String,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = CardBackground)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Filled.Delete,
                    contentDescription = null,
                    tint = AccentPink,
                    modifier = Modifier.size(48.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "Delete Preset?",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Are you sure you want to delete \"$presetName\"?",
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 14.sp
                )

                Spacer(modifier = Modifier.height(20.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color.White
                        )
                    ) {
                        Text("Cancel")
                    }
                    Button(
                        onClick = onConfirm,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = AccentPink,
                            contentColor = Color.White
                        )
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Delete,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Delete")
                    }
                }
            }
        }
    }
}

@Composable
fun InfoChip(label: String, value: Int, color: Color) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = label,
            color = color,
            fontWeight = FontWeight.Bold,
            fontSize = 14.sp
        )
        Text(
            text = "$value",
            color = Color.White,
            fontWeight = FontWeight.Bold,
            fontSize = 18.sp
        )
    }
}

@Composable
fun GradientSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    gradientColors: List<Color>
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.8f)
            )
            Text(
                text = "${value.toInt()}",
                style = MaterialTheme.typography.bodyMedium,
                color = gradientColors.last(),
                fontWeight = FontWeight.Bold
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        Box(modifier = Modifier.fillMaxWidth()) {
            // Track background with gradient
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(
                        Brush.horizontalGradient(gradientColors)
                    )
                    .align(Alignment.Center)
            )
            Slider(
                value = value,
                onValueChange = onValueChange,
                valueRange = 0f..255f,
                modifier = Modifier.fillMaxWidth(),
                colors = SliderDefaults.colors(
                    thumbColor = gradientColors.last(),
                    activeTrackColor = Color.Transparent,
                    inactiveTrackColor = Color.Transparent
                )
            )
        }
    }
}

@Composable
fun CircularLedLayout(
    ledCount: Int,
    selectedLed: Int,
    ledStates: List<LedState>,
    onLedClick: (Int) -> Unit
) {
    Layout(
        content = {
            repeat(ledCount) { index ->
                val state = ledStates[index]
                val ledColor = Color(
                    red = state.red,
                    green = state.green,
                    blue = state.blue,
                    alpha = 255
                )
                val isSelected = index == selectedLed

                // Animated scale for selected LED
                val scale by animateFloatAsState(
                    targetValue = if (isSelected) 1.2f else 1f,
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioMediumBouncy,
                        stiffness = Spring.StiffnessLow
                    ),
                    label = "scale"
                )

                val animatedColor by animateColorAsState(
                    targetValue = ledColor,
                    animationSpec = tween(300),
                    label = "color"
                )

                Box(contentAlignment = Alignment.Center) {
                    // Glow effect for each LED
                    if (isSelected || (state.red + state.green + state.blue) > 100) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .blur(8.dp)
                                .background(animatedColor.copy(alpha = 0.5f), CircleShape)
                        )
                    }
                    Box(
                        modifier = Modifier
                            .size(32.dp)
                            .scale(scale)
                            .shadow(if (isSelected) 8.dp else 2.dp, CircleShape)
                            .clip(CircleShape)
                            .background(
                                Brush.radialGradient(
                                    colors = listOf(
                                        animatedColor,
                                        animatedColor.copy(alpha = 0.7f)
                                    )
                                )
                            )
                            .border(
                                width = if (isSelected) 2.dp else 1.dp,
                                brush = if (isSelected) Brush.linearGradient(
                                    colors = listOf(AccentCyan, AccentPurple)
                                ) else Brush.linearGradient(
                                    colors = listOf(Color.White.copy(alpha = 0.3f), Color.White.copy(alpha = 0.1f))
                                ),
                                shape = CircleShape
                            )
                            .clickable(
                                interactionSource = remember { MutableInteractionSource() },
                                indication = null
                            ) { onLedClick(index) },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "$index",
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            color = if ((state.red + state.green + state.blue) / 3 > 128)
                                Color.Black.copy(alpha = 0.8f) else Color.White
                        )
                    }
                }
            }
        }
    ) { measurables, constraints ->
        val placeables = measurables.map { it.measure(constraints) }
        val radius = minOf(constraints.maxWidth, constraints.maxHeight) / 2 - 25
        val centerX = constraints.maxWidth / 2
        val centerY = constraints.maxHeight / 2

        layout(constraints.maxWidth, constraints.maxHeight) {
            placeables.forEachIndexed { index, placeable ->
                val angle = (2 * Math.PI * index / ledCount) - Math.PI / 2
                val x = centerX + (radius * cos(angle)).toInt() - placeable.width / 2
                val y = centerY + (radius * sin(angle)).toInt() - placeable.height / 2
                placeable.placeRelative(x, y)
            }
        }
    }
}

