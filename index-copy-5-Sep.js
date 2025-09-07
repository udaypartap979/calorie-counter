// ============================================================================
// index.js  —  Single-file, structured layout (no logic changes)
// ============================================================================
//
// TABLE OF CONTENTS
//  0) Strict mode
//  1) Environment & Imports
//  2) App Initialization & Middleware
//  3) User Profile & Device Bias (ENV-driven)
//  4) Prompts (Food / Food-Image / Workout) + Classifiers
//  5) Third-Party Clients & Configs (OpenAI, Gemini, Anthropic, Multer, Mail)
//  6) Utility & Helper Functions (RNNoise, audio preprocess, Whisper, analyzers,
//     media download, cravings, formatting replies, Supabase logging helper,
//     WhatsApp audio processing helper)
//  7) API Endpoints (Priority): /whatsapp-webhook, /analyze-* , /log-analysis,
//     /handle-craving
//  8) API Endpoints (Secondary, commented): Spoonacular paths etc. (APPENDIX)
//  9) Server Root & Listener
//
// Notes:
// - Code/logic untouched; only reorganized and heavily commented.
// - Comment banners (====) make scanning easier.
// ============================================================================

"use strict"; // 0) safer defaults without altering behavior

// =========================
// 1) ENVIRONMENT & IMPORTS
// =========================
require("dotenv").config();

const express   = require("express");
const cors      = require("cors");
const multer    = require("multer");
const axios     = require("axios");
const OpenAI    = require("openai");
const fs        = require("fs");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const nodemailer = require("nodemailer");
const supabase   = require("./supabaseClient");
const twilio     = require("twilio");
const path       = require("path");
const https      = require("https");
const { exec }   = require("child_process");
const util       = require("util");
const execPromise = util.promisify(exec);
const Anthropic   = require("@anthropic-ai/sdk");
const FormData  = require("form-data");
const morgan    = require("morgan");
const QuickChart = require("quickchart-js");
const cron = require("node-cron");
const fetch = (...args) => import("node-fetch").then(({ default: fetch }) => fetch(...args));
// add near top of file with other requires
const multerLib = require("multer");
const uploadNone = multerLib().none(); // use this for parsing Exotel callback form-data fields






// ==========================================
// 2) APP INITIALIZATION & GLOBAL MIDDLEWARE
// ==========================================
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(express.urlencoded({ extended: true }));
app.use(morgan("combined")); // request logs
app.use(express.urlencoded({ extended: true })); // 👈 add this


const port = process.env.PORT || 3000;

// Whisper prompts (env-provided so you can tune transcription behavior)
const WHISPER_SYSTEM_PROMPT  = process.env.WHISPER_SYSTEM_PROMPT;
const WHISPER_CONTEXT_PROMPT = process.env.WHISPER_CONTEXT_PROMPT;

// Optional RNNoise model (for audio denoising via ffmpeg arnndn). If missing, we fall back gracefully.
const RNNOISE_MODEL = process.env.RNNOISE_MODEL_PATH;
// Backup idea (hosted model) left as a note in original: const RNNOISE_URL = process.env.RNNOISE_MODEL_URL;

// ==================================================
// 3) USER PROFILE & DEVICE BIAS (ENV-DRIVEN DEFAULT)
// ==================================================
const USER_WEIGHT_KG     = Number(process.env.USER_WEIGHT_KG || 70);
const USER_AGE           = Number(process.env.USER_AGE || 30);
const USER_SEX           = (process.env.USER_SEX || "unknown").toLowerCase();
const APPLE_WATCH_ADJUST = Number(process.env.APPLE_WATCH_ADJUST || 1.0);

// ====================================
// 4) PROMPTS + CLASSIFIER HELPERS
// ====================================

// ------ FOOD PARSER (text) ------
const SYS_FOOD_TEXT = `
You are a nutrition facts engine. Extract foods and nutrition ONLY from trusted sources:
- Brand label
- Open Food Facts (packaged foods)
- IFCT (India)
- USDA FDC
- Nutritionix

STRICT RULES:
- For each food item, ALWAYS include an "assumptions" array at the item level.
  Examples:
  ["assumed medium chapati = 40 g", "assumed 1 katori dal = 200 ml"]
- Assumptions must describe how you inferred portion sizes, serving sizes, or database choices.
- If no assumption is needed (exact DB match with explicit portion), still return ["exact match from USDA"].
- Totals must also include an "assumptions" array summarizing overall reasoning.
- Never invent values; if uncertain, return 0 calories, confidence 0, source "unknown".
- Confidence = 1.0 for exact DB match, ≤0.8 for inferred portions.
- Output strict JSON only.
`;


const USER_FOOD_TEXT = (content) => `
Extract foods, portion, and nutrition from this text. 

Use ONLY values from Brand labels, Open Food Facts, IFCT, USDA, or Nutritionix. 
State which source was used in "source". If multiple foods, return one entry per food.

Return ONLY:
{
  "type": "food",
  "details": [
    {
      "item": "string (with portion assumption if inferred)",
      "quantity": number,
      "unit": "string",
      "calories": number,
      "macros": { "protein": number, "fat": number, "carbs": number },
      "brand": "string",
      "source": "string",
      "confidence": number,
      "assumptions": ["string", ...]
    }
  ],
  "totals": {
    "calories": number,
    "assumptions": ["string", ...],
    "confidence": number
  }
}

TEXT:
"""${content}"""
`;



// ------ IMAGE FOOD PARSER ------
const SYS_FOOD_IMAGE = `
You are a vision nutrition parser. Use Brand labels, Open Food Facts, IFCT, USDA, or Nutritionix.

STRICT RULES:
- Only include visible portion (plate, bowl, serving), not entire dish.
- For each food item, ALWAYS include an "assumptions" array at the item level.
  Examples:
  ["assumed 1 plate rice ≈ 150 g", "assumed small bowl of dal ≈ 200 ml"]
- If a label is visible, assumptions = ["nutrition taken directly from label"].
- Totals must also include an "assumptions" array summarizing overall reasoning.
- Confidence = 1.0 if label or DB exact match, ≤0.8 if portion inferred.
- Output strict JSON only.
`;




const USER_FOOD_IMAGE = `
Analyze this image (and caption if provided) and return ONLY:
{
  "type": "food",
  "details": [
    {
      "item": "string (with portion assumption if inferred)",
      "quantity": number,
      "unit": "string",
      "calories": number,
      "macros": { "protein": number, "fat": number, "carbs": number },
      "brand": "string",
      "source": "string",
      "confidence": number,
      "assumptions": ["string", ...]
    }
  ],
  "totals": {
    "calories": number,
    "assumptions": ["string", ...],
    "confidence": number
  }
}

Rules:
- Use Brand label if available, otherwise use OFF/IFCT/USDA/Nutritionix.
- Portion = visible serving (plate, bowl, piece).
- Explicitly state assumptions in "assumptions".
- If nothing matches reliably, return empty details.
`;



// ------ WORKOUT ESTIMATOR (system prompt + user prompt builder) ------
const SYS_WORKOUT_ESTIMATOR = `
You are an exercise energy–expenditure estimator.

Steps:
1. Parse input for activities, durations, intensity, distance, pace, incline, resistance, HR clues.
2. Use Compendium of Physical Activities (MET values) or closest equivalent.
3. If intensity unclear, pick the lowest reasonable MET (to avoid overestimation).
4. Compute calories with: kcal_per_min = MET * 3.5 * weight_kg / 200; total = kcal_per_min * duration_minutes.
5. Multiply by APPLE_WATCH_ADJUST if provided.
6. Always round calories to whole numbers.

STRICT RULES:
- Always state assumptions (e.g., "assumed jogging pace 8 km/h").
- Confidence reflects input quality:
  * exact duration + intensity given → ≥ 0.8
  * inferred values → ≤ 0.5
- Never return 0 kcal if duration > 0.
- If no workout is detected: return { "type":"workout", "details":[], "totals":{"calories_burned":0,"assumptions":["no workout found"],"confidence":0.0} }

Output must be strict JSON:
{
  "type": "workout",
  "details": [...],
  "totals": { "calories_burned": number, "assumptions": ["string", ...], "confidence": number }
}
`;


function buildWorkoutUserPrompt({ modality, text, imageHint }) {
  return `
Context:
- Modality: ${modality}
- User profile: { "weight_kg": ${USER_WEIGHT_KG}, "age": ${USER_AGE}, "sex": "${USER_SEX}" }
- Device bias: { "APPLE_WATCH_ADJUST": ${APPLE_WATCH_ADJUST} }

Input:
"""
${text || imageHint || ""}
"""

Tasks:
1) Extract each activity and its duration (minutes). Parse any intensity/distance/pace/incline/resistance/HR cues.
2) Choose a reasonable MET for each activity and estimate calories using the system rules.
3) Return ONLY the JSON schema defined in the system prompt.
`.trim();
}

// ------ UNIVERSAL CLASSIFIERS (food vs workout) ------
async function classifyFoodOrWorkoutFromText(text) {
  try {
    const r = await openai.chat.completions.create({
      model: OPENAI_MODEL_TEXT,
      temperature: 0,
      messages: [
        { role: "system", content: "Classify the input as 'food' or 'workout'. Return only one word." },
        { role: "user", content: `INPUT:\n"""${text}"""` }
      ]
    });
    return (r.choices?.[0]?.message?.content || "food").trim().toLowerCase().includes("workout") ? "workout" : "food";
  } catch {
    return "food";
  }
}

async function classifyFoodOrWorkoutFromImage(imageBase64, mimeType) {
  try {
    const r = await openai.chat.completions.create({
      model: OPENAI_MODEL_VISION,
      temperature: 0,
      messages: [
        { role: "system", content: "Look at the image and return only one word: 'food' or 'workout'." },
        { role: "user", content: [
          { type: "text", text: "Classify this image as food or workout. Return only one word." },
          { type: "image_url", image_url: { url: `data:${mimeType};base64,${imageBase64}` } }
        ]}
      ]
    });
    return (r.choices?.[0]?.message?.content || "food").trim().toLowerCase().includes("workout") ? "workout" : "food";
  } catch {
    return "food";
  }
}

// =======================================================
// 5) THIRD-PARTY CLIENTS & BASIC CONFIG (instances etc.)
// =======================================================
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const genAI  = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const GEMINI_MODEL        = process.env.GEMINI_MODEL        || "gemini-2.5-pro";
const geminiModel         = genAI.getGenerativeModel({ model: GEMINI_MODEL });
const anthropic           = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const OPENAI_MODEL_VISION = process.env.OPENAI_MODEL_VISION || "gpt-4o";
const OPENAI_MODEL_TEXT   = process.env.OPENAI_MODEL_TEXT   || "gpt-4-turbo";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
});

const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "gurjeetchem@gmail.com",
    pass: "vsvb ltyz eqfp wleu",
  },
});

const twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
const BASE_URL     = process.env.BASE_URL || `http://localhost:${port}`; // public URL or local

// ======================================================================
// 6) UTILITIES & HELPERS (audio pipeline, analyzers, logging, etc.)
// ======================================================================

// -- RNNoise existence check (optional denoiser) --
async function ensureRnnoiseModel() {
  if (RNNOISE_MODEL && fs.existsSync(RNNOISE_MODEL)) {
    return RNNOISE_MODEL;
  }
  return null; // run without arnndn if no model present
}

// -- ffmpeg audio preprocess (denoise, trim silence, normalize) --
async function preprocessAudio(rawPath, cleanedPath) {
  const modelPath = await ensureRnnoiseModel();
  const hasRnnoise = !!(modelPath && fs.existsSync(modelPath));
  const denoise = hasRnnoise ? `arnndn=m=${modelPath},` : "";
  const af = `"${denoise}silenceremove=1:0:-50dB,loudnorm=i=-22:tp=-2:lra=7"`;
  const cmd = `ffmpeg -y -hide_banner -loglevel error -i "${rawPath}" -af ${af} -ac 1 -ar 16000 "${cleanedPath}"`;
 
  try {
    await execPromise(cmd);
    return cleanedPath;
  } catch (e) {
    console.warn("[transcribe] Preprocess failed, using raw audio. Reason:", e.message);
    return rawPath;
  }
}

// -- Whisper transcription (uses env prompts) --
async function transcribeAudioWithWhisper(audioBuffer) {
  const stamp = Date.now();
  const rawPath = `/tmp/${stamp}-raw.audio`;     // let ffmpeg probe; extension doesn’t matter
const cleanedPath = `/tmp/${stamp}-cleaned.wav`; // safe, linear PCM for Whisper

  let pathForWhisper = rawPath;
  let stream = null;

  fs.writeFileSync(rawPath, audioBuffer);

  try {
    await ensureRnnoiseModel();
    pathForWhisper = await preprocessAudio(rawPath, cleanedPath);

    if (!fs.existsSync(pathForWhisper)) {
      throw new Error(`Audio file missing before upload: ${pathForWhisper}`);
    }
    stream = fs.createReadStream(pathForWhisper);

    const response = await openai.audio.transcriptions.create({
      file: stream,
      model: "whisper-1",
     // temperature: 0,
     // language: "en",
      prompt: `${WHISPER_CONTEXT_PROMPT}`,
    });

    console.log("[Whisper] transcription OK", response);
    return response.text ?? response;
  } catch (err) {
    console.error("Error transcribing with Whisper:", err);
    throw new Error("Failed to transcribe audio.");
  } finally {
    try { if (stream) await new Promise((res) => stream.close(res)); } catch {}
    for (const p of [rawPath, cleanedPath]) {
      try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {}
    }
  }
}

// -- Text analyzer (food/workout) --
async function analyzeContentWithChatGPT(content) {
  const cls = await classifyFoodOrWorkoutFromText(content);

  if (cls === "workout") {
    const resp = await openai.chat.completions.create({
      model: OPENAI_MODEL_TEXT,
      temperature: 0.1,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYS_WORKOUT_ESTIMATOR },
        { role: "user", content: buildWorkoutUserPrompt({ modality: "text", text: content }) }
      ]
    });
    let out = {};
    try { out = JSON.parse(resp.choices[0].message.content || "{}"); } catch {}
    if (!out || out.type !== "workout") out = { type: "workout", details: [], totals: { calories_burned: 0, assumptions: ["fallback"], confidence: 0 } };
    return out;
  }

  const foodResp = await openai.chat.completions.create({
    model: OPENAI_MODEL_TEXT,
    temperature: 0.2,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: SYS_FOOD_TEXT },
      { role: "user", content: USER_FOOD_TEXT(content) }
    ]
  });

  let out = {};
  try { out = JSON.parse(foodResp.choices[0].message.content || "{}"); } catch {}
  if (!out || out.type !== "food") return { type: "food", details: [] };

  if (!Array.isArray(out.details)) out.details = [];
  out.details = out.details.map(d => ({
    item: d?.item ?? "",
    quantity: Number(d?.quantity ?? 0) || 0,
    unit: d?.unit ?? "",
    calories: Number(d?.calories ?? 0) || 0,
    macros: {
      protein: Number(d?.macros?.protein ?? 0) || 0,
      fat:     Number(d?.macros?.fat ?? 0) || 0,
      carbs:   Number(d?.macros?.carbs ?? 0) || 0,
    },
    brand: d?.brand ?? "",
    source: d?.source ?? "",
    confidence: Math.max(0, Math.min(1, Number(d?.confidence ?? 0))) || 0
  }));

  return out;
}

// -- Audio analyzer (transcribe -> detect both food + workout) --
async function analyzeAudioWithGPT(audioBuffer) {
  const transcript = await transcribeAudioWithWhisper(audioBuffer);
  console.log("[Audio->Transcript]", transcript);

  // 🟢 New: Ask GPT to split into BOTH food + workout
  const prompt = `
Transcript: """${transcript}"""

Task:
- Extract food mentions (items, portions, nutrition) using same schema as your food parser.
- Extract workout mentions (activities, durations, calories burned) using same schema as your workout parser.
- If one is missing, return it as empty.

Return strict JSON:
{
  "food": {
    "type": "food",
    "details": [...],
    "totals": { "calories": number, "assumptions": ["string"], "confidence": number }
  },
  "workout": {
    "type": "workout",
    "details": [...],
    "totals": { "calories_burned": number, "assumptions": ["string"], "confidence": number }
  },
  "transcript": "${transcript}"
}
`;

  const resp = await openai.chat.completions.create({
    model: OPENAI_MODEL_TEXT,
    temperature: 0.2,
    messages: [{ role: "system", content: prompt }]
  });

  let out = {};
  try {
    out = JSON.parse(resp.choices[0].message.content || "{}");
  } catch {
    out = {
      food: { type: "food", details: [], totals: { calories: 0, assumptions: ["parse failed"], confidence: 0 } },
      workout: { type: "workout", details: [], totals: { calories_burned: 0, assumptions: ["parse failed"], confidence: 0 } },
      transcript
    };
  }

  console.log("[Analysis] food items:", out.food?.details?.length, "workout items:", out.workout?.details?.length);
  return out;
}


// -- Image analyzer (vision path; also supports workout machines) --
async function analyzeImageWithChatGPT(imageBuffer, mimeType) {
  const imageBase64 = imageBuffer.toString("base64");
  const cls = await classifyFoodOrWorkoutFromImage(imageBase64, mimeType);

  if (cls === "workout") {
    const resp = await openai.chat.completions.create({
      model: OPENAI_MODEL_VISION,
      temperature: 0.1,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYS_WORKOUT_ESTIMATOR },
        {
          role: "user",
          content: [
            { type: "text", text: buildWorkoutUserPrompt({ modality: "image", imageHint: "Read any machine console text (time, pace, distance, kcal). If kcal not shown, estimate via system rules." }) },
            { type: "image_url", image_url: { url: `data:${mimeType};base64,${imageBase64}` } }
          ]
        }
      ]
    });
    let out = {};
    try { out = JSON.parse(resp.choices[0].message.content || "{}"); } catch {}
    if (!out || out.type !== "workout") out = { type: "workout", details: [], totals: { calories_burned: 0, assumptions: ["fallback"], confidence: 0 } };
    return out;
  }

  const foodResp = await openai.chat.completions.create({
    model: OPENAI_MODEL_VISION,
    temperature: 0.2,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: SYS_FOOD_IMAGE },
      {
        role: "user",
        content: [
          { type: "text", text: USER_FOOD_IMAGE },
          { type: "image_url", image_url: { url: `data:${mimeType};base64,${imageBase64}` } }
        ]
      }
    ]
  });

  let out = {};
  try { out = JSON.parse(foodResp.choices[0].message.content || "{}"); } catch {}
  if (!out || out.type !== "food") return { type: "food", details: [] };

  if (!Array.isArray(out.details)) out.details = [];
  out.details = out.details.map(d => ({
    item: d?.item ?? "",
    quantity: Number(d?.quantity ?? 0) || 0,
    unit: d?.unit ?? "",
    calories: Number(d?.calories ?? 0) || 0,
    macros: {
      protein: Number(d?.macros?.protein ?? 0) || 0,
      fat:     Number(d?.macros?.fat     ?? 0) || 0,
      carbs:   Number(d?.macros?.carbs   ?? 0) || 0,
    },
    brand: d?.brand ?? "",
    source: d?.source ?? "",
    confidence: Math.max(0, Math.min(1, Number(d?.confidence ?? 0))) || 0
  }));

  return out;
}

// -- Twilio media downloader (auth to Twilio CDN) --
async function downloadTwilioMedia(mediaUrl) {
  const response = await axios({
    method: "get",
    url: mediaUrl,
    responseType: "arraybuffer",
    auth: {
      username: process.env.TWILIO_ACCOUNT_SID,
      password: process.env.TWILIO_AUTH_TOKEN,
    },
  });
  return Buffer.from(response.data);
}

// -- Craving coach text (WhatsApp-friendly formatting) --
async function getCravingResponseFromGPT(cravingText) {
  const prompt = `
    You are a helpful and empathetic nutritional coach. The user is telling you about a food craving.
    Your task is to provide a supportive and informative response.

    1.  First, identify the specific food the user is craving from their message.
    2.  Provide 3-4 practical, actionable tips to help them overcome this specific craving. Examples include drinking a glass of water, going for a short walk, or suggesting a healthier alternative.
    3.  Next, briefly list 2-3 potential negative health effects of consuming this food in excess. Keep the tone factual and non-judgmental.
    4.  Format the entire response for WhatsApp, using asterisks for bolding (e.g., *Tip 1:*). Do not use markdown like hashes (#).
    5.  If you cannot identify a specific food, provide general advice for managing cravings.

    User's message: "${cravingText}"

    Example response for a pizza craving:
    It sounds like you're craving pizza right now. Here are a few things you can try to manage that craving:

    *Stay Hydrated:* Sometimes our body mistakes thirst for hunger. Try drinking a full glass of water and wait 15 minutes.

    *Go for a Walk:* A short, 10-15 minute walk can help distract you and reset your mind.

    *Opt for a Healthier Alternative:* If you really want those flavors, try a whole-wheat pita with tomato sauce, a sprinkle of cheese, and your favorite veggies.

    *A quick note on pizza:* While a slice can be a nice treat, eating it often can lead to high sodium intake and a surplus of calories from refined carbs, which might leave you feeling sluggish later.
  `;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
    });
    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error getting craving response from GPT:", error);
    throw new Error("Failed to get craving response.");
  }
}

// -- Helper to process WhatsApp audio end-to-end and then log it --
async function processAndLogAudio(audioBuffer, userIdentifier, mime = "application/octet-stream", filename = "whatsapp-audio") {
  // Build a real Node.js multipart form
  const form = new FormData();
  // Preserve extension hint if we know the mime
  const ext = mime.includes("ogg") ? ".ogg"
            : mime.includes("aac") ? ".aac"
            : mime.includes("mpeg") ? ".mp3"
            : "";
  form.append("audio", audioBuffer, { filename: `${filename}${ext}`, contentType: mime });

  // Call analyze-audio
  const { data: analysis } = await axios.post(
    `${BASE_URL}/analyze-audio`,
    form,
    { headers: form.getHeaders() }
  );
  if (!analysis) throw new Error("Could not analyze the audio.");

  // Log to /log-analysis (optional: attach the raw audio too)
  const logForm = new FormData();
  logForm.append("userId", userIdentifier);
  logForm.append("userEmail", "xyz@gmail.com");
  logForm.append("analysisResult", JSON.stringify(analysis));
  logForm.append("audio", audioBuffer, { filename: `${filename}${ext}`, contentType: mime });

  await axios.post(`${BASE_URL}/log-analysis`, logForm, { headers: logForm.getHeaders() });
  return analysis;
}


// -- Format a concise WhatsApp reply from analysis JSON --
const buildReplyForAnalysis = (analysis) => {
  if (!analysis || !analysis.type) {
    return "Sorry, I couldn't understand that.";
  }

  if (analysis.type === "workout") {
    const details = Array.isArray(analysis.details) ? analysis.details : [];
    const total = details.reduce((s, d) => s + (Number(d.calories_burned) || 0), 0);

    const lines = details.map((d) => {
      const act = d?.activity || "Activity";
      const mins = Number(d?.duration_min || 0);
      const kcal = Math.round(Number(d?.calories_burned || 0));
      const intensity = (d?.intensity && d.intensity !== "unknown") ? `, ${d.intensity}` : "";
      const conf = d?.confidence != null ? ` (conf: ${d.confidence})` : "";
      const assumptions = d?.assumptions?.length ? `\n   assumptions: ${d.assumptions.join("; ")}` : "";
      return `• ${act}${intensity} — ${mins} min ≈ ${kcal} kcal${conf}${assumptions}`;
    });

    const header = "Workout:";
    const summary = `Total estimated calories: ${Math.round(total)}.`;
    return `${header}\n${lines.join("\n")}\n${summary}`;
  }

  // FOOD
  const details = Array.isArray(analysis.details) ? analysis.details : [];

  let totalCalories = 0;
  let totalProtein = 0;
  let totalFat = 0;
  let totalCarbs = 0;

  const lines = details.map((i) => {
    const name = i?.item || "Item";
    const qty = Number(i?.quantity || 0);
    const unit = i?.unit || "";
    const kcal = Math.round(Number(i?.calories || 0));
    const protein = Math.round(Number(i?.macros?.protein || 0));
    const fat     = Math.round(Number(i?.macros?.fat || 0));
    const carbs   = Math.round(Number(i?.macros?.carbs || 0));

    totalCalories += kcal;
    totalProtein += protein;
    totalFat += fat;
    totalCarbs += carbs;

    const portion = (qty && unit) ? ` (${qty} ${unit})` : (qty ? ` (${qty})` : "");

    return `• ${name}${portion} — ${kcal} kcal, P:${protein}g, F:${fat}g, C:${carbs}g`;
  });

  const header = "Food:";
  const summary = `Totals — Calories: ${totalCalories}, Protein: ${totalProtein}g, Fat: ${totalFat}g, Carbs: ${totalCarbs}g`;

  const reference = `\n\n*Healthy Daily Reference* (average adult):\nCalories: ~2000 kcal\nProtein: ~75g\nFat: ~65g\nCarbs: ~250g`;

  if (lines.length) {
    return `${header}\n${lines.join("\n")}\n${summary}${reference}`;
  }
  return `${header}\n${summary}${reference}`;

};


// Coerce "250", "250 kcal", 250 -> 250; fallback to def if NaN
function num(x, def = 0) {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string") {
    // strip non-numeric (keep digits, decimal, sign, exponent)
    const cleaned = x.replace(/[^0-9eE+.\-]/g, "");
    const n = Number(cleaned);
    return Number.isFinite(n) ? n : def;
  }
  return def;
}


// -- Supabase logging helper used by /log-analysis --
/* (kept inside route in your original; here we leave route logic intact and
   compute totals/confidence in-place there to avoid moving behavior) */

// ===================================
// 7) API ENDPOINTS (PRIMARY PATHS)
// ===================================

app.get("/whatsapp-webhook", (req, res) => {
  const VERIFY_TOKEN = process.env.META_VERIFY_TOKEN;
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    console.log("✅ Meta webhook verified");
    res.status(200).send(challenge); // echo back challenge
  } else {
    res.sendStatus(403);
  }
});

/**
 * Flow: WhatsApp user message -> Twilio -> /whatsapp-webhook
 * Routes to image/audio/text analyzers, logs, then replies.
 */
app.post("/whatsapp-webhook", async (req, res) => {
  const incomingMsg = req.body;
  const userPhoneNumber = incomingMsg.From;
  let responseMessage = "Processing your request...";

  // 1) Immediate ack to the user (so WhatsApp shows quick feedback)
  await twilioClient.messages.create({
    from: incomingMsg.To,
    to: userPhoneNumber,
    body: responseMessage,
  });

  // 2) Acknowledge Twilio's webhook
  res.status(200).send();

  // ---------- local helpers used only within this route ----------
  const logAnalysis = async ({ analysis, mediaBuffer, contentType }) => {
    const logFormData = new FormData();

if (mediaBuffer) {
  logFormData.append("image", mediaBuffer, {
    filename: "log-image.jpg",
    contentType: contentType || "image/jpeg",
  });
}

logFormData.append("userId", userPhoneNumber);
logFormData.append("userEmail", "xyz@gmail.com");
logFormData.append("analysisResult", JSON.stringify(analysis));

await axios.post(`${BASE_URL}/log-analysis`, logFormData, {
  headers: logFormData.getHeaders(),
});
  };

  try {
    // ---------- (A) MEDIA MESSAGE ----------
    if (incomingMsg.MediaContentType0 && incomingMsg.MediaUrl0) {
      const mediaUrl = incomingMsg.MediaUrl0;
      const contentType = incomingMsg.MediaContentType0 || "";
      const mediaBuffer = await downloadTwilioMedia(mediaUrl);

      let analysis = null;

      if (contentType.includes("image")) {
        const formData = new FormData();
        formData.append("image", mediaBuffer, { filename: "whatsapp-image.jpg", contentType });

        if (incomingMsg.Body) {
          formData.append("text", incomingMsg.Body); // attach caption text
        }

        const { data } = await axios.post(`${BASE_URL}/analyze-image-with-text`, formData, {
          headers: formData.getHeaders(),
        });

        analysis = data;

        await logAnalysis({ analysis, mediaBuffer, contentType });

      } else if (contentType.includes("audio")) {
        analysis = await processAndLogAudio(
          mediaBuffer,
          userPhoneNumber,
          contentType,
          "whatsapp-audio"
        );
        
      } else {
        responseMessage = "Food/Workout: Unsupported media type. Please send an image or audio.";
        await twilioClient.messages.create({
          from: incomingMsg.To,
          to: userPhoneNumber,
          body: responseMessage,
        });
        return;
      }

      // 3) Final reply
      responseMessage = buildReplyForAnalysis(analysis);
      console.log("[WhatsApp Reply] ->", responseMessage);

    }

    // ---------- (B) TEXT MESSAGE ----------
    else if (incomingMsg.Body) {
      const userText = incomingMsg.Body;
      const lowerCaseText = userText.toLowerCase();
    
      if (lowerCaseText.includes("crave") || lowerCaseText.includes("craving")) {
        // existing craving logic
        const { data } = await axios.post(`${BASE_URL}/handle-craving`, { text: userText });
        responseMessage = data.advice;
    
      } else if (
        lowerCaseText.includes("what should i eat") || 
        lowerCaseText.includes("diet suggestion") ||
        lowerCaseText.includes("recommend food")
      ) {
        // 👇 new diet recommendation logic
        const { data } = await axios.post(`${BASE_URL}/recommend-food`, {
          userId: userPhoneNumber // or however you store user ID
        });
        responseMessage = data.recommendations;
    
      } 
      else if (lowerCaseText.includes("analyze")) {
        const { data } = await axios.post(`${BASE_URL}/analyze-summary`, {
          userId: userPhoneNumber,
        });
      
        await twilioClient.messages.create({
          from: incomingMsg.To,
          to: userPhoneNumber,
          body: "📊 Here’s your Calories Summary",
          mediaUrl: [data.caloriesChartUrl],
        });
      
        await twilioClient.messages.create({
          from: incomingMsg.To,
          to: userPhoneNumber,
          body: "📊 Here’s your Macros Summary",
          mediaUrl: [data.macrosChartUrl],
        });
      
        return; // stop further processing
      }     
      else {
        // default food/workout analyzer
        const { data: analysis } = await axios.post(`${BASE_URL}/analyze-text`, { text: userText });
        await logAnalysis({ analysis });
        responseMessage = buildReplyForAnalysis(analysis);
      }
    }
    

  } catch (error) {
    console.error("Error processing WhatsApp message:", error);
    responseMessage = "Food/Workout: Sorry, I couldn't process that. Please try again.";
  }

  // 3) Send the final, detailed response
  //console.log("[WHATSAPP] incoming audio:", { contentType: mediaContentType, bytes: mediaBuffer.length });

  const quoted = incomingMsg.Body || (incomingMsg.MediaContentType0 ? "[image/audio]" : "");
const finalReply = quoted
  ? `*You said:* \n> ${quoted}\n\n${responseMessage}`
  : responseMessage;

await twilioClient.messages.create({
  from: incomingMsg.To,
  to: userPhoneNumber,
  body: finalReply,
});
  /*await twilioClient.messages.create({
    from: incomingMsg.To,
    to: userPhoneNumber,
    body: responseMessage,
    context: {
      message_id: incomingMsg.SmsMessageSid  // 👈 this makes WhatsApp render inline quote
    }
  });*/
});

// ---------- ANALYZE AUDIO ----------
app.post("/analyze-audio", upload.single("audio"), async (req, res) => {
  try {
    let audioBuffer;

    if (req.file) {
      audioBuffer = req.file.buffer;
    } else if (req.body.url) {
      const audioRes = await fetch(req.body.url);
      audioBuffer = await audioRes.buffer();
    } else {
      return res.status(400).json({ error: "No audio provided." });
    }

    const analysis = await analyzeAudioWithGPT(audioBuffer);

    return res.status(200).json(analysis);
  } catch (e) {
    console.error("Error in /analyze-audio:", e);
    return res.status(500).json({ error: "Failed to analyze audio." });
  }
});


// ---------- ANALYZE TEXT ----------
app.post("/analyze-text", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: "No text provided." });
    }
    const analysis = await analyzeContentWithChatGPT(text);
    res.status(200).json(analysis);
  } catch (error) {
    console.error("Error in /analyze-text endpoint:", error);
    res.status(500).json({ error: error.message });
  }
});

// ---------- ANALYZE IMAGE ----------
app.post("/analyze-image", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file provided." });
    }
    const analysis = await analyzeImageWithChatGPT(req.file.buffer, req.file.mimetype);
    res.status(200).json(analysis);
  } catch (error) {
    console.error("Error in /analyze-image endpoint:", error);
    res.status(500).json({ error: error.message });
  }
});

// ---------- ANALYZE IMAGE + TEXT ----------
app.post("/analyze-image-with-text", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file provided." });
    }

    const captionText = req.body.text || ""; // optional user text
    const imageBase64 = req.file.buffer.toString("base64");
    const mimeType = req.file.mimetype;

    // Ask GPT vision model with both inputs
    const resp = await openai.chat.completions.create({
      model: OPENAI_MODEL_VISION,
      temperature: 0.2,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYS_FOOD_IMAGE },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: captionText
                ? `${USER_FOOD_IMAGE}\n\nExtra context from user: ${captionText}`
                : USER_FOOD_IMAGE
            },
            {
              type: "image_url",
              image_url: { url: `data:${mimeType};base64,${imageBase64}` }
            }
          ]
        }
      ]
    });

    let out = {};
    try {
      out = JSON.parse(resp.choices[0].message.content || "{}");
    } catch {
      out = { type: "food", details: [] };
    }

    if (!Array.isArray(out.details)) out.details = [];
    out.details = out.details.map((d) => ({
      item: d?.item ?? "",
      quantity: Number(d?.quantity ?? 0) || 0,
      unit: d?.unit ?? "",
      calories: Number(d?.calories ?? 0) || 0,
      macros: {
        protein: Number(d?.macros?.protein ?? 0) || 0,
        fat: Number(d?.macros?.fat ?? 0) || 0,
        carbs: Number(d?.macros?.carbs ?? 0) || 0,
      },
      brand: d?.brand ?? "",
      source: d?.source ?? "",
      confidence: Math.max(0, Math.min(1, Number(d?.confidence ?? 0))) || 0
    }));

    res.status(200).json(out);
  } catch (error) {
    console.error("Error in /analyze-image-with-text endpoint:", error);
    res.status(500).json({ error: "Failed to analyze image with text." });
  }
});


// ---------- LOG ANALYSIS (to Supabase; optionally with image) ----------
app.post("/log-analysis", upload.fields([{ name: "image" }, { name: "audio" }]), async (req, res) => {
  try {
    const { analysisResult, userId, userEmail } = req.body;

    if (!analysisResult || !userId || !userEmail) {
      return res.status(400).json({
        error: "analysisResult, userId, and userEmail are required.",
      });
    }

    let parsedAnalysis;
try {
  parsedAnalysis = typeof analysisResult === "string" ? JSON.parse(analysisResult) : analysisResult;
} catch {
  return res.status(400).json({ error: "analysisResult must be valid JSON." });
}

    const { type = "food", details } = parsedAnalysis;
    let imageUrl = null;

    if (req.files?.image?.[0]) {
      const f = req.files.image[0];
      const fileName = `${Date.now()}-${f.originalname}`;
      await supabase.storage
        .from("meal-images")
        .upload(fileName, f.buffer, { contentType: f.mimetype });
      const { data: urlData } = supabase.storage
        .from("meal-images")
        .getPublicUrl(fileName);
      imageUrl = urlData.publicUrl;
    }

    let audioUrl = null;

    if (req.files?.audio?.[0]) {
      const a = req.files.audio[0];
      const fileName = `${Date.now()}-${a.originalname}`;
      await supabase.storage
        .from("audio-notes") // 👈 use this bucket
        .upload(fileName, a.buffer, { contentType: a.mimetype || "audio/mpeg" });

      const { data: urlData } = supabase.storage
        .from("audio-notes")
        .getPublicUrl(fileName);
      audioUrl = urlData.publicUrl;
    }

    
    let totalCalories = 0;
let overallConfidence = null;

// FOOD totals: use details (not items), and coerce "250 kcal" → 250
if (type === "food" && Array.isArray(parsedAnalysis.details)) {
  totalCalories = parsedAnalysis.details.reduce((sum, it) => {
    return sum + num(it?.calories, 0);
  }, 0);

  const vals = parsedAnalysis.details
    .map(d => (typeof d?.confidence === "number" ? d.confidence : null))
    .filter(v => v != null);

  overallConfidence = vals.length
    ? +(vals.reduce((a,b)=>a+b,0) / vals.length).toFixed(3)
    : null;

// WORKOUT totals: same idea, but from calories_burned
} else if (type === "workout" && Array.isArray(parsedAnalysis.details)) {
  totalCalories = parsedAnalysis.details.reduce((sum, d) => {
    return sum + num(d?.calories_burned, 0);
  }, 0);
}

    
    const evalGemini = parsedAnalysis._eval?.gemini || null;
    const evalClaude = parsedAnalysis._eval?.claude || null;
    
    function averageConfidence(items) {
      const vals = items
        .map(it => (it.calorie_estimate && typeof it.calorie_estimate.confidence === "number") ? it.calorie_estimate.confidence : null)
        .filter(v => v != null);
      if (!vals.length) return null;
      return +(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3);
    }
    
    const newLog = {
      user_id: userId,
      user_email: userEmail,
      item_type: type,
      total_calories: Math.round(totalCalories),
      log_details: parsedAnalysis,
      image_url: imageUrl,
      audio_url: audioUrl,
      ai_confidence: overallConfidence,
      eval_gemini: evalGemini,
      eval_claude: evalClaude
    };

    const { data, error } = await supabase.from("meals").insert([newLog]).select();

    if (error) {
      console.error("Supabase insert error:", error);
      throw error;
    }

    res.status(201).json({
      message: "Analysis logged successfully!",
      data: data,
    });
  } catch (error) {
    console.error("Error in /log-analysis endpoint:", error);
    res.status(500).json({ error: "Failed to log analysis." });
  }
});

// ---------- HANDLE CRAVING ----------
// ---------- HANDLE CRAVING ----------
app.post("/handle-craving", async (req, res) => {
  try {
    const { text, userId } = req.body; // 👈 pass userId in webhook call
    if (!text) {
      return res.status(400).json({ error: "No text provided." });
    }

    // === Existing craving advice logic (UNCHANGED) ===
    const advice = await getCravingResponseFromGPT(text);

    // === NEW: Fetch today's + week's logs ===
    let todayTotals = { calories: 0, protein: 0, fat: 0, carbs: 0 };
    let weekTotals = { calories: 0, protein: 0, fat: 0, carbs: 0 };

    if (userId) {
      const { data: meals, error } = await supabase
        .from("meals")
        .select("*")
        .eq("user_id", userId)
        .gte("created_at", new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()); // last 7 days

      if (meals && !error) {
        const todayStr = new Date().toISOString().split("T")[0];

        for (const meal of meals) {
          const d = new Date(meal.created_at).toISOString().split("T")[0];
          const totals = {
            calories: meal.total_calories || 0,
            protein: meal.log_details?.totals?.protein || 0,
            fat: meal.log_details?.totals?.fat || 0,
            carbs: meal.log_details?.totals?.carbs || 0,
          };

          weekTotals.calories += totals.calories;
          weekTotals.protein += totals.protein;
          weekTotals.fat += totals.fat;
          weekTotals.carbs += totals.carbs;

          if (d === todayStr) {
            todayTotals.calories += totals.calories;
            todayTotals.protein += totals.protein;
            todayTotals.fat += totals.fat;
            todayTotals.carbs += totals.carbs;
          }
        }
      }
    }

    // === NEW: GPT-based negative impact analysis ===
    const negativePrompt = `
The user has a craving: "${text}".

Today's totals so far:
Calories: ${todayTotals.calories}, Protein: ${todayTotals.protein} g, Fat: ${todayTotals.fat} g, Carbs: ${todayTotals.carbs} g

This week's totals so far:
Calories: ${weekTotals.calories}, Protein: ${weekTotals.protein} g, Fat: ${weekTotals.fat} g, Carbs: ${weekTotals.carbs} g

Explain in 2-3 WhatsApp-friendly bullet points what the negative health impact would be if the user gives into this craving today, based on their intake so far. Keep tone factual, not judgmental.
    `;

    let negativeImpact = "";
    try {
      const negResp = await openai.chat.completions.create({
        model: OPENAI_MODEL_TEXT,
        temperature: 0.5,
        messages: [
          { role: "system", content: "You are a factual nutrition assistant." },
          { role: "user", content: negativePrompt }
        ]
      });
      negativeImpact = negResp.choices[0].message.content || "";
    } catch (err) {
      console.error("Error getting negative impact GPT response:", err);
    }

    // === Merge advice + negative impact ===
    const finalAdvice = advice + (negativeImpact ? `\n\n*Potential impact if you give in:*\n${negativeImpact}` : "");

    res.status(200).json({ advice: finalAdvice });

  } catch (error) {
    console.error("Error in /handle-craving endpoint:", error);
    res.status(500).json({ error: error.message });
  }
});


app.post("/recommend-food", async (req, res) => {
  try {
    const { userId } = req.body;
    if (!userId) {
      return res.status(400).json({ error: "userId required" });
    }

    // 1. Fetch meals from Supabase
    const { data: meals, error } = await supabase
      .from("meals")
      .select("*")
      .eq("user_id", userId)
      .gte("created_at", new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()); // last 7 days

    if (error) throw error;

    // 2. Aggregate macros
    let todayTotals = { calories: 0, protein: 0, fat: 0, carbs: 0 };
    let weekTotals = { calories: 0, protein: 0, fat: 0, carbs: 0 };

    const todayStr = new Date().toISOString().split("T")[0];

    for (const meal of meals) {
      const d = new Date(meal.created_at).toISOString().split("T")[0];
      const totals = {
        calories: meal.total_calories || 0,
        protein: meal.log_details?.totals?.protein || 0,
        fat: meal.log_details?.totals?.fat || 0,
        carbs: meal.log_details?.totals?.carbs || 0,
      };

      weekTotals.calories += totals.calories;
      weekTotals.protein += totals.protein;
      weekTotals.fat += totals.fat;
      weekTotals.carbs += totals.carbs;

      if (d === todayStr) {
        todayTotals.calories += totals.calories;
        todayTotals.protein += totals.protein;
        todayTotals.fat += totals.fat;
        todayTotals.carbs += totals.carbs;
      }
    }

    // 3. Healthy targets (can be personalized later)
    const dailyTarget = { calories: 2000, protein: 75, fat: 65, carbs: 250 };

    // 4. Gaps
    const gaps = {
      calories: dailyTarget.calories - todayTotals.calories,
      protein: dailyTarget.protein - todayTotals.protein,
      fat: dailyTarget.fat - todayTotals.fat,
      carbs: dailyTarget.carbs - todayTotals.carbs,
    };

    // 5. Build GPT prompt
    const prompt = `
    You are a diet coach. Based on today's intake and weekly balance, suggest foods.
    
    Today's totals so far:
    Calories: ${todayTotals.calories}
    Protein: ${todayTotals.protein} g
    Fat: ${todayTotals.fat} g
    Carbs: ${todayTotals.carbs} g
    
    Target (daily):
    Calories: ${dailyTarget.calories}
    Protein: ${dailyTarget.protein} g
    Fat: ${dailyTarget.fat} g
    Carbs: ${dailyTarget.carbs} g
    
    Gaps remaining today:
    Calories: ${gaps.calories}
    Protein: ${gaps.protein}
    Fat: ${gaps.fat}
    Carbs: ${gaps.carbs}
    
    Instructions:
    - Suggest 3 Indian food options and 3 non-Indian options that help fill the gaps.
    - Each option must include:
      • Food + portion size
      • Approximate nutrition (calories + macros)
      • A short explanation: *why this helps* (e.g., "high in protein to cover today's deficit").
    - Explanations must be specific to today's and this week's totals.
    - Keep tone factual and WhatsApp-friendly.
    - Output plain text.
    - Keep explanations for each option under 15 words.
    - Do not repeat macros if similar to another option.

        `;
    

    const resp = await openai.chat.completions.create({
      model: OPENAI_MODEL_TEXT,
      temperature: 0.6,
      messages: [
        { role: "system", content: "You are a dietitian that gives healthy food suggestions." },
        { role: "user", content: prompt },
      ],
    });

    res.status(200).json({
      todayTotals,
      weekTotals,
      gaps,
      recommendations: resp.choices[0].message.content,
    });
  } catch (err) {
    console.error("Error in /recommend-food:", err);
    res.status(500).json({ error: "Failed to generate food recommendations." });
  }
});

app.post("/analyze-summary", async (req, res) => {
  try {
    const { userId } = req.body;
    if (!userId) return res.status(400).json({ error: "userId required" });

    // Fetch last 7 days meals
    const { data: meals, error } = await supabase
      .from("meals")
      .select("*")
      .eq("user_id", userId)
      .gte("created_at", new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString());

    if (error) throw error;

    // Aggregate
    const todayStr = new Date().toISOString().split("T")[0];
    let today = { calories: 0, protein: 0, fat: 0, carbs: 0};
    let week = { calories: 0, protein: 0, fat: 0, carbs: 0};

    for (const meal of meals) {
      const d = new Date(meal.created_at).toISOString().split("T")[0];
      let protein = meal.log_details?.totals?.protein || 0;
let fat     = meal.log_details?.totals?.fat || 0;
let carbs   = meal.log_details?.totals?.carbs || 0;

// If macros not present at totals level, sum them from details
if ((!protein && !fat && !carbs) && Array.isArray(meal.log_details?.details)) {
  meal.log_details.details.forEach(it => {
    protein += Number(it?.macros?.protein || 0);
    fat     += Number(it?.macros?.fat || 0);
    carbs   += Number(it?.macros?.carbs || 0);
  });
}

const totals = {
  calories: meal.total_calories || meal.log_details?.totals?.calories || 0,
  protein,
  fat,
  carbs
};


      week.calories += totals.calories;
      week.protein += totals.protein;
      week.fat += totals.fat;
      week.carbs += totals.carbs;

      if (d === todayStr) {
        today.calories += totals.calories;
        today.protein += totals.protein;
        today.fat += totals.fat;
        today.carbs += totals.carbs;
      }
    }

    // --- Chart 1: Calories ---
    const caloriesChart = new QuickChart();
    caloriesChart.setConfig({
      type: "bar",
      data: {
        labels: ["Calories"],
        datasets: [
          {
            label: "Today",
            data: [today.calories],
            backgroundColor: "rgba(54, 162, 235, 0.6)",
          },
          {
            label: "This Week",
            data: [week.calories],
            backgroundColor: "rgba(255, 99, 132, 0.6)",
          },
        ],
      },
      options: {
        title: { display: true, text: "Calories Summary" },
        responsive: true,
        scales: { y: { beginAtZero: true } },
      },
    });
    const caloriesChartUrl = caloriesChart.getUrl();

    // --- Chart 2: Macros ---
    const macrosChart = new QuickChart();
    macrosChart.setConfig({
      type: "bar",
      data: {
        labels: ["Protein", "Fat", "Carbs"],
        datasets: [
          {
            label: "Today",
            data: [today.protein, today.fat, today.carbs],
            backgroundColor: "rgba(54, 162, 235, 0.6)",
          },
          {
            label: "This Week",
            data: [week.protein, week.fat, week.carbs],
            backgroundColor: "rgba(255, 99, 132, 0.6)",
          },
        ],
      },
      options: {
        title: { display: true, text: "Macros Summary" },
        responsive: true,
        scales: { y: { beginAtZero: true } },
      },
    });
    const macrosChartUrl = macrosChart.getUrl();

    res.status(200).json({
      today,
      week,
      caloriesChartUrl,
      macrosChartUrl,
    });
  } catch (err) {
    console.error("Error in /analyze-summary:", err);
    res.status(500).json({ error: "Failed to analyze summary" });
  }
});

// -----------------------------
// triggerExotelCall (replace existing)
// -----------------------------
async function triggerExotelCall(opts = {}) {
  // opts may contain overrides: { from, exophone, sid, flowId, record, apiKey, apiToken, statusCallbackBase }
  const sid      = opts.sid || process.env.EXOTEL_SID;
  const apiKey   = opts.apiKey || process.env.EXOTEL_API_KEY;
  const apiToken = opts.apiToken || process.env.EXOTEL_API_TOKEN;
  const exophone = opts.exophone || process.env.EXOPHONE;
  const from     = opts.from || process.env.MY_PHONE;
  const flowId   = opts.flowId || process.env.EXOTEL_FLOW_ID || "1065544";
  const baseUrl  = (opts.statusCallbackBase || process.env.BASE_URL || `http://localhost:${port}`).replace(/\/$/, "");

  if (!sid || !apiKey || !apiToken || !exophone || !from) {
    const missing = [];
    if (!sid) missing.push("EXOTEL_SID");
    if (!apiKey) missing.push("EXOTEL_API_KEY");
    if (!apiToken) missing.push("EXOTEL_API_TOKEN");
    if (!exophone) missing.push("EXOPHONE");
    if (!from) missing.push("MY_PHONE");
    const err = new Error("Missing Exotel credentials/env: " + missing.join(", "));
    console.error(err.message);
    throw err;
  }

  try {
    console.log("Manual trigger received, calling triggerExotelCall()");

    // Exotel connect endpoint (use axios auth rather than embedding credentials in URL)
    const url = `https://api.exotel.com/v1/Accounts/${sid}/Calls/connect`;

    // Build form body. Keep Record:false for Passthru approach so the <Record> returned by start-recording runs.
    const payload = new URLSearchParams({
      From: from,
      CallerId: exophone,
      Url: `http://my.exotel.com/${sid}/exoml/start_voice/${flowId}`,
      StatusCallback: `${baseUrl}/exotel-callback`,
      Record: "true"
    });

    const response = await axios.post(url, payload.toString(), {
      auth: { username: apiKey, password: apiToken },
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      timeout: 30000
    });

    console.log("Exotel Call Triggered:", response.data);
    return { ok: true, status: response.status, data: response.data };
  } catch (err) {
    const respBody = err?.response?.data || err?.message;
    console.error("Error triggering Exotel call:", respBody);
    const error = new Error("Exotel trigger failed");
    error.details = respBody;
    error.status = err?.response?.status || 500;
    throw error;
  }
}



// -----------------------------
// /simulate-trigger route that calls triggerExotelCall()
// -----------------------------
app.post("/simulate-trigger", express.json(), async (req, res) => {
  try {
    // Optional overrides from body or query
    const overrides = {
      from: req.body?.from || req.query?.from,
      exophone: req.body?.exophone || req.query?.exophone,
      sid: req.body?.sid || req.query?.sid,
      flowId: req.body?.flowId || req.query?.flowId,
      record: (req.body?.record === true) || (req.query?.record === "true"),
      apiKey: req.body?.apiKey || req.query?.apiKey,
      apiToken: req.body?.apiToken || req.query?.apiToken,
      statusCallbackBase: req.body?.statusCallbackBase || req.query?.statusCallbackBase
    };

    // Call the trigger function (this will perform the actual Exotel API POST)
    const result = await triggerExotelCall(overrides);

    return res.status(200).json({
      ok: true,
      message: "triggerExotelCall executed",
      result
    });
  } catch (err) {
    console.error("simulate-trigger error:", err.details ?? err.message);
    return res.status(err.status || 500).json({
      ok: false,
      message: "Failed to trigger Exotel call",
      error: err.details || err.message
    });
  }
});

// -------------------
// /start-recording route — returns ExoML (TwiML-like) telling Exotel to record
// -------------------
app.all("/start-recording", (req, res) => {
  console.log("PASSTHRU -> /start-recording called:", {
    method: req.method,
    query: req.query,
    body: req.body,
  });

  // Public callback endpoint (must be your ngrok/BASE_URL)
  const callbackUrl = `${process.env.BASE_URL.replace(/\/$/, "")}/exotel-callback`;
  console.log("callbackUrl", callbackUrl);
  // Exotel interprets this XML (similar to Twilio TwiML)
  // <Record> will:
  //  - play a beep
  //  - record up to 120s
  //  - stop on "#"
  //  - POST the file info to /exotel-callback
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="female">Please tell me your food and workout after the beep. Press hash when finished.</Say>
  <Record playBeep="true" maxLength="120" finishOnKey="#" action="${callbackUrl}" method="POST" />
  <Say>We did not receive any recording. Goodbye.</Say>
</Response>`;

  res.set("Content-Type", "text/xml");
  res.status(200).send(xml);
});

// -------------------
// /exotel-callback route — receives Exotel's POST and forwards recording to analyze endpoint
// Uses uploadNone (multer().none()) to parse multipart/form-data fields reliably
// -------------------
app.post("/exotel-callback", uploadNone, async (req, res) => {
  try {
    // multer.none() (uploadNone) or urlencoded parser will have populated req.body
    console.log("🔔 /exotel-callback fields:", req.body);
    console.log("🔔 /exotel-callback fields:", JSON.stringify(req.body || {}, null, 2));
console.log("🔔 keys:", Object.keys(req.body || {}));


    const fields = req.body || {};

    // Try common keys Exotel may use
    const RecordingUrl =
      fields.RecordingUrl ||
      fields.recording_url ||
      fields.recordingUrl ||
      (Array.isArray(fields.RecordingUrl) && fields.RecordingUrl[0]) ||
      null;

    const Status = fields.Status || fields.status || fields.CallStatus || null;

    // ACK quickly so Exotel won't retry
    res.sendStatus(200);

    if (!RecordingUrl) {
      console.warn("No RecordingUrl in callback — fields were:", fields);
      return;
    }

    console.log("✅ Got RecordingUrl:", RecordingUrl, "Status:", Status, " — downloading...");

    // Download the recording
    let audioResp;
    try {
      audioResp = await axios.get(RecordingUrl, { responseType: "arraybuffer", timeout: 60000 });
    } catch (err) {
      console.error("Failed to download recording URL:", err.message || err);
      return;
    }

    if (audioResp.status !== 200) {
      console.error("Non-200 when downloading recording:", audioResp.status);
      return;
    }

    const audioBuffer = Buffer.from(audioResp.data);

    // forward as multipart/form-data to analyze-audio
    const form = new FormData();
    form.append("audio", audioBuffer, {
      filename: "exotel_recording.mp3",
      contentType: audioResp.headers["content-type"] || "audio/mpeg",
    });

    console.log("Forwarding recording to /analyze-audio...");
    try {
      const forwardResp = await axios.post(`${process.env.BASE_URL.replace(/\/$/, "")}/analyze-audio`, form, {
        headers: form.getHeaders(),
        timeout: 120000,
      });
      console.log("analyze-audio response status:", forwardResp.status);
    } catch (err) {
      console.error("Error forwarding to analyze-audio:", (err.response && err.response.data) || err.message);
    }

  } catch (err) {
    console.error("Unexpected error in /exotel-callback:", err);
    try { res.sendStatus(500); } catch (e) {}
  }
});


// -------------------- SCHEDULER --------------------
// Runs every day at 9PM IST
cron.schedule("0 21 * * *", () => {
  console.log("9PM reached → Triggering Exotel call...");
  triggerExotelCall();
});
// =========================
// 5. SPOONACULAR API HELPERS (SECONDARY)
// =========================

/**
 * Flow: Button: '🔍 Analyze Photo' -> handleAnalyzeImage (frontend) -> /identify-food (backend endpoint) -> searchFoodSpoonacular (current function)
 * @param {string} foodName
 * @returns {Promise<object|null>} The first search result or null.
 * Searches for a food ingredient using Spoonacular API.
 * Calls: axios.get (Spoonacular API)
 * Called by: getFoodInfoSpoonacular (index.js)
 * Indirectly called by: handleAnalyzeImage, handleLogMeal, handleLogAudio in calorie-frontend/src/app/page.js via /identify-food, /log-meal, /log-audio endpoints
 * Triggered by: "🔍 Analyze Photo", "✔ Log This Meal", "✔ Log Voice Note" buttons in page.js
 */
/*
async function searchFoodSpoonacular(foodName) {
  try 
    const response = await axios.get(
      `https://api.spoonacular.com/food/ingredients/search`,
      {
        params: {
          query: foodName,
          number: 1,
          apiKey: process.env.SPOONACULAR_API_KEY,
        },
      }
    );
    if (response.data.results && response.data.results.length > 0) {
      return response.data.results[0];
    }
    return null;
  } catch (error) {
    console.error(
      `Error searching food in Spoonacular: ${foodName}:`,
      error.message
    );
    return null;
  }
}
*/
/**
 * Flow: Button: '🔍 Analyze Photo' -> handleAnalyzeImage (frontend) -> /identify-food (backend endpoint) -> getFoodInfoSpoonacular -> getNutritionInfoSpoonacular (current function)
 * @param {number} ingredientId - The Spoonacular ID for the ingredient.
 * @param {number} amount - The amount of the ingredient.
 * @param {string} unit - The unit for the amount (e.g., "grams").
 * @returns {Promise<object|null>} Detailed nutrition object or null.
 * Called by: getFoodInfoSpoonacular (index.js)
 * Indirectly called by: handleAnalyzeImage, handleLogMeal, handleLogAudio in calorie-frontend/src/app/page.js via /identify-food, /log-meal, /log-audio endpoints
 * Triggered by: "🔍 Analyze Photo", "✔ Log This Meal", "✔ Log Voice Note" buttons in page.js
 * Not called directly from frontend (page.js)
 */
/*
async function getNutritionInfoSpoonacular(
  ingredientId,
  amount = 100,
  unit = "grams"
) {
  try {
    const response = await axios.get(
      `https://api.spoonacular.com/food/ingredients/${ingredientId}/information`,
      {
        params: {
          amount: amount,
          unit: unit,
          apiKey: process.env.SPOONACULAR_API_KEY,
        },
      }
    );
    if (response.data && response.data.nutrition) {
      const nutrition = response.data.nutrition;
      const calories = nutrition.nutrients.find((n) => n.name === "Calories");
      return {
        name: response.data.name,
        calories: calories ? Math.round(calories.amount) : 0,
        serving_size: `${amount} ${unit}`,
        weight_grams: amount,
        nutrition: {
          protein:
            nutrition.nutrients.find((n) => n.name === "Protein")?.amount || 0,
          fat: nutrition.nutrients.find((n) => n.name === "Fat")?.amount || 0,
          carbs:
            nutrition.nutrients.find((n) => n.name === "Carbohydrates")
              ?.amount || 0,
          fiber:
            nutrition.nutrients.find((n) => n.name === "Fiber")?.amount || 0,
          sugar:
            nutrition.nutrients.find((n) => n.name === "Sugar")?.amount || 0,
        },
      };
    }
    return null;
  } catch (error) {
    console.error(
      `Error getting nutrition info from Spoonacular:`,
      error.message
    );
    return null;
  }
}
*/
/**
 * Flow: Button: '🔍 Analyze Photo' -> handleAnalyzeImage (frontend) -> /identify-food (backend endpoint) -> getFoodInfoSpoonacular -> searchRecipeSpoonacular (current function)
 * @param {string} dishName - The name of the dish.
 * @returns {Promise<object|null>} Recipe nutrition object or null.
 * Searches for a recipe and its nutrition data using Spoonacular.
 * Called by: getFoodInfoSpoonacular (index.js)
 * Indirectly called by: handleAnalyzeImage, handleLogMeal, handleLogAudio in calorie-frontend/src/app/page.js via /identify-food, /log-meal, /log-audio endpoints
 * Triggered by: "🔍 Analyze Photo", "✔ Log This Meal", "✔ Log Voice Note" buttons in page.js
 * Not called directly from frontend (page.js)
 */
/*
async function searchRecipeSpoonacular(dishName) {
  try {
    const response = await axios.get(
      `https://api.spoonacular.com/recipes/complexSearch`,
      {
        params: {
          query: dishName,
          number: 1,
          addRecipeNutrition: true,
          apiKey: process.env.SPOONACULAR_API_KEY,
        },
      }
    );
    if (response.data.results && response.data.results.length > 0) {
      const recipe = response.data.results[0];
      const nutrition = recipe.nutrition;
      if (nutrition && nutrition.nutrients) {
        const calories = nutrition.nutrients.find((n) => n.name === "Calories");
        return {
          name: recipe.title,
          calories: calories ? Math.round(calories.amount) : 0,
          serving_size: `1 serving (${recipe.servings} total servings)`,
          servings: recipe.servings,
          nutrition: {
            protein:
              nutrition.nutrients.find((n) => n.name === "Protein")?.amount ||
              0,
            fat: nutrition.nutrients.find((n) => n.name === "Fat")?.amount || 0,
            carbs:
              nutrition.nutrients.find((n) => n.name === "Carbohydrates")
                ?.amount || 0,
            fiber:
              nutrition.nutrients.find((n) => n.name === "Fiber")?.amount || 0,
            sugar:
              nutrition.nutrients.find((n) => n.name === "Sugar")?.amount || 0,
          },
          source: "recipe",
        };
      }
    }
    return null;
  } catch (error) {
    console.error(
      `Error searching recipe in Spoonacular: ${dishName}:`,
      error.message
    );
    return null;
  }
}
*/
/**
 * Flow: Button: '🔍 Analyze Photo' -> handleAnalyzeImage (frontend) -> /identify-food (backend endpoint) -> getFoodInfoSpoonacular (current function)
 * A comprehensive function to get food info, trying ingredients first, then recipes.
 * @param {string} foodName - The name of the food.
 * @returns {Promise<object|null>} The best available nutrition info or null.
 * Called by: /identify-food, /log-audio, /log-meal endpoints (index.js)
 * Indirectly called by: handleAnalyzeImage, handleLogMeal, handleLogAudio in calorie-frontend/src/app/page.js
 * Triggered by: "🔍 Analyze Photo", "✔ Log This Meal", "✔ Log Voice Note" buttons in page.js
 */
/*
async function getFoodInfoSpoonacular(foodName) {
  try {
    // First, try to find a matching ingredient
    const ingredient = await searchFoodSpoonacular(foodName);
    if (ingredient) {
      const nutritionInfo = await getNutritionInfoSpoonacular(
        ingredient.id,
        100,
        "grams"
      );
      if (nutritionInfo) {
        return {
          ...nutritionInfo,
          source: "ingredient",
        };
      }
    }
    const recipeInfo = await searchRecipeSpoonacular(foodName);
    if (recipeInfo) {
      return recipeInfo;
    }
    return null;
  } catch (error) {
    console.error(
      `Error getting food info from Spoonacular: ${foodName}:`,
      error.message
    );
    return null;
  }
}
*/
/**
 * Flow: Button: '🔍 Analyze Photo' -> handleAnalyzeImage (frontend) -> /identify-food (backend endpoint) -> getQuickNutritionGuess (current function)
 * @param {string} foodName - The name of the food.
 * @returns {Promise<object|null>} A quick nutrition guess or null.
 * Gets a quick nutrition guess for a food using Spoonacular's guessNutrition endpoint.
 * Called by: /identify-food endpoint, fallback in getFoodInfoSpoonacular (index.js)
 * Indirectly called by: handleAnalyzeImage in calorie-frontend/src/app/page.js
 * Triggered by: "🔍 Analyze Photo" button in page.js
 * Not called directly from frontend (page.js)
 */
/*
async function getQuickNutritionGuess(foodName) {
  try {
    const response = await axios.get(
      `https://api.spoonacular.com/recipes/guessNutrition`,
      {
        params: {
          title: foodName,
          apiKey: process.env.SPOONACULAR_API_KEY,
        },
      }
    );
    if (response.data && response.data.calories) {
      return {
        name: foodName,
        calories: Math.round(response.data.calories.value),
        serving_size: "estimated portion",
        nutrition: {
          protein: Math.round(response.data.protein?.value || 0),
          fat: Math.round(response.data.fat?.value || 0),
          carbs: Math.round(response.data.carbs?.value || 0),
        },
        source: "nutrition_guess",
      };
    }
    return null;
  } catch (error) {
    console.error(
      `Error getting nutrition guess from Spoonacular: ${foodName}:`,
      error.message
    );
    return null;
  }
}
*/
// =========================
// 6. API ENDPOINTS (SECONDARY)
// =========================

/**
 * POST /identify-food
 * Identifies food items in an uploaded image using Gemini, then fetches nutrition info for each item.
 * Calls: geminiModel.generateContent, getFoodInfoSpoonacular, getQuickNutritionGuess
 * Called by: handleAnalyzeImage in calorie-frontend/src/app/page.js
 * Triggered by: "🔍 Analyze Photo" button in page.js
 */
/*
app.post("/identify-food", upload.single("foodImage"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file provided." });
    }
    const prompt = `
Analyze this image and identify all distinct, edible food items and drinks.
- For composite dishes (like 'Chicken and Waffles'), identify the main dish name.
- For separate items (like drinks or side sauces), list them individually.
- Exclude all non-edible items like plates, cutlery, tablecloths, or people.
- Return the list as a simple comma-separated string.
- Example output: Fried Chicken, Waffle, Syrup, Butter
`;
    const imagePart = {
      inlineData: {
        data: req.file.buffer.toString("base64"),
        mimeType: req.file.mimetype,
      },
    };
    const result = await geminiModel.generateContent([prompt, imagePart]);
    const geminiResponseText = result.response.text();
    const identifiedFoods = geminiResponseText
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item);
    if (identifiedFoods.length === 0) {
      return res
        .status(404)
        .json({ error: "No food items could be identified." });
    }
    const foodsWithNutrition = await Promise.all(
      identifiedFoods.map(async (food) => {
        let nutritionInfo =
          (await getFoodInfoSpoonacular(food)) ||
          (await getQuickNutritionGuess(food));
        if (nutritionInfo) {
          return {
            name: nutritionInfo.name,
            calories: nutritionInfo.calories,
            serving_size: nutritionInfo.serving_size,
            nutrition: nutritionInfo.nutrition,
            source: nutritionInfo.source,
          };
        } else {
          return {
            name: food,
            calories: "Unknown",
            nutrition: null,
            source: "not_found",
          };
        }
      })
    );
    const totalCalories = foodsWithNutrition.reduce(
      (sum, food) => sum + (Number(food.calories) || 0),
      0
    );
    res.status(200).json({
      identifiedFoods: foodsWithNutrition,
      totalEstimatedCalories: totalCalories,
      note: "Food identification by Gemini 1.5 Pro. Nutrition values are estimates provided by Spoonacular API.",
    });
  } catch (error) {
    console.error("ERROR during image analysis:", error);
    res.status(500).json({ error: "Failed to analyze image." });
  }
});
*/
/**
 * POST /log-meal
 * Logs a meal with image and nutrition analysis to Supabase.
 * Calls: supabase.storage.upload, supabase.from('meals').insert
 * Called by: handleLogMeal in calorie-frontend/src/app/page.js
 * Triggered by: "✔ Log This Meal" button in page.js
 */
/*
app.post("/log-meal", upload.single("foodImage"), async (req, res) => {
  try {
    const { userId, userEmail, analysisResult } = req.body;
    if (!req.file || !userId || !analysisResult) {
      return res
        .status(400)
        .json({ error: "Image, User ID, and analysis result are required." });
    }
    const fileName = `${Date.now()}-${req.file.originalname}`;
    await supabase.storage
      .from("meal-images")
      .upload(fileName, req.file.buffer, { contentType: req.file.mimetype });
    const { data: urlData } = supabase.storage
      .from("meal-images")
      .getPublicUrl(fileName);
    const imageUrl = urlData.publicUrl;
    const parsedAnalysis = JSON.parse(analysisResult);
    const { identifiedFoods, totalEstimatedCalories } = parsedAnalysis;
    const { data, error } = await supabase.from("meals").insert([
      {
        user_id: userId,
        user_email: userEmail,
        image_url: imageUrl,
        total_calories: totalEstimatedCalories,
        item_type: "food",
        log_details: identifiedFoods,
      },
    ]);
    if (error) {
      console.error("Supabase insert error:", error);
      throw error;
    }
    res.status(201).json({ message: "Meal logged successfully!", data });
  } catch (error) {
    console.error("Error in /log-meal endpoint:", error);
    res.status(500).json({ error: "Failed to log meal." });
  }
});
*/
/**
 * POST /log-audio
 * Logs a meal or workout from an audio file, classifies and transcribes using Gemini, then logs to Supabase.
 * Calls: geminiModel.generateContent, getFoodInfoSpoonacular, supabase.from('meals').insert
 * Called by: handleLogAudio in calorie-frontend/src/app/page.js
 * Triggered by: "✔ Log Voice Note" button in page.js
 */
/*
app.post("/log-audio", upload.single("foodAudio"), async (req, res) => {
  try {
    const { userId, userEmail } = req.body;
    const audioFile = req.file;
    if (!audioFile || !userId) {
      return res
        .status(400)
        .json({ error: "Audio file and User ID are required." });
    }
    const audioPart = {
      inlineData: {
        data: audioFile.buffer.toString("base64"),
        mimeType: audioFile.mimetype,
      },
    };
    const classificationPrompt = `Does this audio describe eating food, nutrition, or calories, OR does it describe physical exercise like running, lifting weights, or a workout? Respond with only the word "food" or "workout".`;
    const classificationResult = await geminiModel.generateContent([
      classificationPrompt,
      audioPart,
    ]);
    const itemType = classificationResult.response.text().trim().toLowerCase();
    const transcribeResult = await geminiModel.generateContent([
      "Transcribe this audio.",
      audioPart,
    ]);
    const transcript = transcribeResult.response.text();
    let logDetails = {};
    let totalCalories = 0;
    if (itemType === "workout") {
      const caloriePrompt = `Based on the following workout transcript, provide a rough estimate of the total calories burned. Respond with only a single number. For example: 350. Transcript: "${transcript}"`;
      const calorieResult = await geminiModel.generateContent(caloriePrompt);
      const estimatedCalories =
        parseInt(calorieResult.response.text().trim()) || 0;
      totalCalories = estimatedCalories;
      logDetails = {
        transcript: transcript,
        estimated_calories_burned: estimatedCalories,
      };
    } else {
      const foodPrompt = `From the following text, extract food items and their portion sizes. Respond with a comma-separated list. Text: "${transcript}"`;
      const foodResult = await geminiModel.generateContent(foodPrompt);
      const foodListFromAudio = foodResult.response
        .text()
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      const foodsWithNutrition = await Promise.all(
        foodListFromAudio.map((food) =>
          getFoodInfoSpoonacular(food).then(
            (info) => info || { name: food, calories: 0 }
          )
        )
      );
      logDetails = foodsWithNutrition;
      totalCalories = foodsWithNutrition.reduce(
        (sum, food) => sum + (Number(food.calories) || 0),
        0
      );
    }
    const { error } = await supabase.from("meals").insert([
      {
        user_id: userId,
        user_email: userEmail,
        item_type: itemType,
        total_calories: totalCalories,
        log_details: logDetails,
      },
    ]);
    if (error) throw error;
    res.status(201).json({ message: `${itemType} logged successfully!` });
  } catch (error) {
    console.error("Error in /log-audio endpoint:", error);
    res.status(500).json({ error: "Failed to log audio." });
  }
});
*/
/**
 * GET /meals
 * Fetches all meals for a user from Supabase.
 * Calls: supabase.from('meals').select
 * Not called from frontend (page.js); used by dashboard or other clients
 */
/*
app.get("/meals", async (req, res) => {
  try {
    const { userId } = req.query;
    if (!userId) {
      return res.status(400).json({ error: "User ID is required." });
    }
    const { data, error } = await supabase
      .from("meals")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false });
    if (error) throw error;
    res.status(200).json(data);
  } catch (error) {
    console.error("Error fetching meals:", error);
    res.status(500).json({ error: "Failed to fetch meals." });
  }
});
*/
/**
 * POST /invite-dashboard-access
 * Sends an email invitation to view a user's dashboard.
 * Calls: transporter.sendMail
 * Not called from frontend (page.js)
 */
/*
app.post("/invite-dashboard-access", async (req, res) => {
  const { recipientEmail, userId } = req.body;

  if (!recipientEmail) {
    return res.status(400).json({ error: "Recipient email is required." });
  }

  if (!userId) {
    return res.status(400).json({ error: "User ID is required." });
  }

  const dashboardUrl = `https://calorie-frontend.vercel.app/dashboard?userId=${userId}`;
  const mailOptions = {
    to: recipientEmail,
    subject: "You've been invited to view a dashboard!",
    html: `
<p>Hello,</p>
<p>A friend has invited you to view their personal dashboard.</p>
<p>You can see all their latest activity by visiting this link: <a href="${dashboardUrl}">View Dashboard</a></p>
<p>Best regards,</p>
<p>The Dashboard Team</p>
`,
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log(`Invitation email sent successfully to ${recipientEmail} for userId: ${userId}`);
    res.status(200).json({
      message: "Invitation email sent successfully.",
      dashboardUrl: dashboardUrl,
    });
  } catch (error) {
    console.error("Error sending email:", error);
    res.status(500).json({ error: "Failed to send invitation email." });
  }
  
});
*/
// =========================
// 9. SERVER ROOT & LISTENER
// =========================
/**
 * GET /
 * Health check endpoint for the server.
 * Not called from frontend (page.js)
 */
app.get("/", (req, res) => {
  res.status(200).json({ status: "healthy", message: "Service is running" });
});

/**
 * Starts the Express server and listens on the configured port.
 * Not called from frontend (page.js)
 */
app.listen(port, () => {
  console.log(`✅ Server is running on http://localhost:${port}`);
});