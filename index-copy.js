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


// ==========================================
// 2) APP INITIALIZATION & GLOBAL MIDDLEWARE
// ==========================================
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(morgan("combined")); // request logs

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
You are a precise nutrition parser for Indian and international foods.
Return a compact JSON describing foods from user input. Do not add commentary.
If unsure, fill fields with sensible defaults ("" for strings, 0 for numbers).
`;

const USER_FOOD_TEXT = (content) => `
Extract foods, portion, and basic nutrition from this text.

Return ONLY:
{
  "type": "food",
  "details": [
    {
      "item": "string",
      "quantity": number,
      "unit": "string",
      "calories": number,
      "macros": { "protein": number, "fat": number, "carbs": number },
      "brand": "string",
      "source": "string",
      "confidence": number
    }
  ]
}

Rules:
- Always include quantity & unit (infer if needed).
- "source" must be explicit (e.g., "USDA FDC", "IFCT (India)", "Nutritionix", "Brand label").
- Confidence 0..1.

TEXT:
"""${content}"""
`;

// ------ IMAGE FOOD PARSER ------
const SYS_FOOD_IMAGE = `
You are a vision nutrition parser. Read any visible text/labels.
Infer item names, portions, and rough nutrition. Return compact JSON only.
`;

const USER_FOOD_IMAGE = `
Analyze this image and return ONLY:
{
  "type": "food",
  "details": [
    {
      "item": "string",
      "quantity": number,
      "unit": "string",
      "calories": number,
      "macros": { "protein": number, "fat": number, "carbs": number },
      "brand": "string",
      "source": "string",
      "confidence": number
    }
  ]
}

Notes:
- For packaged items use source="Brand label" when readable; else use a common DB (USDA/IFCT/Nutritionix).
- Include at least one item if any food is visible; else return an empty array.
`;

// ------ WORKOUT ESTIMATOR (system prompt + user prompt builder) ------
const SYS_WORKOUT_ESTIMATOR = `
You are an exercise energy–expenditure estimator.

Goals:
- Parse the input (text, audio transcript, or image) to extract activities, durations, and any intensity, distance, pace, incline, resistance, reps/sets, or heart-rate clues.
- Choose a reasonable MET per activity using the Compendium of Physical Activities (or closest equivalent). If intensity is unclear, pick the closest common value (light/moderate/vigorous).
- Compute calories with: kcal_per_min = MET * 3.5 * weight_kg / 200; calories = kcal_per_min * duration_minutes.
- If APPLE_WATCH_ADJUST > 1 (e.g., 1.25) is provided, multiply calories by that factor to match “active calories” style devices.
- Never return 0 calories if duration > 0; provide your best estimate and note assumptions.
- If some inputs are missing, infer sensible defaults and state them in "assumptions".
- Keep estimates conservative when the evidence is weak, and reflect that in "confidence".

Output STRICTLY as JSON (no extra prose) with this shape:
{
  "type": "workout",
  "details": [
    {
      "activity": "string",
      "duration_min": number,
      "intensity": "light|moderate|vigorous|unknown",
      "met": number,
      "calories_burned": number,
      "components": {
        "kcal_per_min": number,
        "unadjusted_total": number,
        "adjusted_factor": number,
        "adjusted_total": number
      },
      "assumptions": ["string", ...],
      "confidence": number
    }
  ],
  "totals": {
    "calories_burned": number,
    "assumptions": ["string", ...],
    "confidence": number
  }
}

Rules:
- Round calories to whole numbers; keep MET and kcal_per_min to 1 decimal when helpful.
- Use SI units; convert if the user gives miles/feet.
- If multiple activities are mentioned, return one detail per activity.
- If no workout is detectable, return {"type":"workout","details":[], "totals":{"calories_burned":0,"assumptions":["no workout found"],"confidence":0.0}}
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

// -- Audio analyzer (transcribe -> same text flow) --
async function analyzeAudioWithGPT(audioBuffer) {
  const transcript = await transcribeAudioWithWhisper(audioBuffer);
  console.log("transcript", transcript.text);
  console.log("[Audio->Transcript] length:", (transcript || "").length, "preview:", (transcript || "").slice(0, 120));

  const cls = await classifyFoodOrWorkoutFromText(transcript);
  console.log("[Classifier] decided:", cls);


  if (cls === "workout") {
    const resp = await openai.chat.completions.create({
      model: OPENAI_MODEL_TEXT,
      temperature: 0.1,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYS_WORKOUT_ESTIMATOR },
        { role: "user", content: buildWorkoutUserPrompt({ modality: "audio_transcript", text: transcript }) }
      ]
    });
    let out = {};
    try { out = JSON.parse(resp.choices[0].message.content || "{}"); } catch {}
    if (!out || out.type !== "workout") out = { type: "workout", details: [], totals: { calories_burned: 0, assumptions: ["fallback"], confidence: 0 } };
    console.log("[Analysis] type:", out?.type, "details_count:", Array.isArray(out?.details) ? out.details.length : 0);
    return out;
  }

  const foodResp = await openai.chat.completions.create({
    model: OPENAI_MODEL_TEXT,
    temperature: 0.2,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: SYS_FOOD_TEXT },
      { role: "user", content: USER_FOOD_TEXT(transcript) }
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


  console.log("[Analysis] type:", out?.type, "details_count:", Array.isArray(out?.details) ? out.details.length : 0);

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
    return "Food/Workout: Sorry, I couldn't understand that.";
  }

  if (analysis.type === "workout") {
    const details = Array.isArray(analysis.details) ? analysis.details : [];
    const total = details.reduce((s, d) => s + (Number(d.calories_burned) || 0), 0);

    const lines = details.map((d) => {
      const act = d?.activity || "Activity";
      const mins = Number(d?.duration_min || 0);
      const kcal = Math.round(Number(d?.calories_burned || 0));
      const intensity = (d?.intensity && d.intensity !== "unknown") ? `, ${d.intensity}` : "";
      return `• ${act}${intensity} — ${mins} min ≈ ${kcal} kcal`;
    });

    const header = "Workout:";
    const summary = `Total estimated calories: ${Math.round(total)}.`;
    if (lines.length) {
      return `${header}\n${lines.join("\n")}\n${summary}`;
    }
    return `${header}\n${summary}`;
  }

  const details = Array.isArray(analysis.details) ? analysis.details : [];
  const total = details.reduce((s, i) => s + (Number(i.calories) || 0), 0);

  const lines = details.map((i) => {
    const name = i?.item || "Item";
    const qty = Number(i?.quantity || 0);
    const unit = i?.unit || "";
    const kcal = Math.round(Number(i?.calories || 0));
    const portion = (qty && unit) ? ` (${qty} ${unit})` : (qty ? ` (${qty})` : "");
    return `• ${name}${portion} — ${kcal} kcal`;
  });

  const header = "Food:";
  const summary = `Total estimated calories: ${Math.round(total)}.`;
  if (lines.length) {
    return `${header}\n${lines.join("\n")}\n${summary}`;
  }
  return `${header}\n${summary}`;
};

// -- Supabase logging helper used by /log-analysis --
/* (kept inside route in your original; here we leave route logic intact and
   compute totals/confidence in-place there to avoid moving behavior) */

// ===================================
// 7) API ENDPOINTS (PRIMARY PATHS)
// ===================================

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
  const logAnalysis = async ({ analysis, mediaBuffer }) => {
    const logFormData = new FormData();

    if (mediaBuffer) {
      const imageBlobForLog = new Blob([mediaBuffer]);
      logFormData.append("image", imageBlobForLog, "log-image.jpg");
    }

    logFormData.append("userId", userPhoneNumber);
    logFormData.append("userEmail", "xyz@gmail.com");
    logFormData.append("analysisResult", JSON.stringify(analysis));

    await axios.post(`${BASE_URL}/log-analysis`, logFormData);
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
        const mediaBlob = new Blob([mediaBuffer], { type: contentType });
        formData.append("image", mediaBlob, "whatsapp-image.jpg");

        const { data } = await axios.post(`${BASE_URL}/analyze-image`, formData);
        analysis = data;

        await logAnalysis({ analysis, mediaBuffer });
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
        const { data } = await axios.post(`${BASE_URL}/handle-craving`, { text: userText });
        responseMessage = data.advice;
      } else {
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

  await twilioClient.messages.create({
    from: incomingMsg.To,
    to: userPhoneNumber,
    body: responseMessage,
  });
});

// ---------- ANALYZE AUDIO ----------
app.post("/analyze-audio", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      console.warn("[/analyze-audio] No file received");
      return res.status(400).json({ error: "No audio file provided." });
    }
    console.log("[/analyze-audio] file received:", { size: req.file.size, mimetype: req.file.mimetype, originalname: req.file.originalname });
    
    const analysis = await analyzeAudioWithGPT(req.file.buffer);
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

// ---------- LOG ANALYSIS (to Supabase; optionally with image) ----------
app.post("/log-analysis", upload.fields([{ name: "image" }, { name: "audio" }]), async (req, res) => {
  try {
    const { analysisResult, userId, userEmail } = req.body;

    if (!analysisResult || !userId || !userEmail) {
      return res.status(400).json({
        error: "analysisResult, userId, and userEmail are required.",
      });
    }

    const parsedAnalysis = JSON.parse(analysisResult);
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
    
    if (type === "food" && Array.isArray(parsedAnalysis.items)) {
      if (type === "food" && Array.isArray(parsedAnalysis.details)) {
        totalCalories = parsedAnalysis.details.reduce((sum, it) => {
          const c = Number(it?.calories || 0);
          return sum + (Number.isFinite(c) ? c : 0);
        }, 0);
        overallConfidence = (() => {
          const vals = parsedAnalysis.details
            .map(d => typeof d?.confidence === "number" ? d.confidence : null)
            .filter(v => v != null);
          return vals.length ? +(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3) : null;
        })();
      }
      
    } else if (type === "workout" && Array.isArray(parsedAnalysis.details)) {
      totalCalories = parsedAnalysis.details.reduce((sum, d) => sum + (Number(d.calories_burned) || 0), 0);
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
app.post("/handle-craving", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: "No text provided." });
    }
    const advice = await getCravingResponseFromGPT(text);
    res.status(200).json({ advice });
  } catch (error) {
    console.error("Error in /handle-craving endpoint:", error);
    res.status(500).json({ error: error.message });
  }
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
  try {
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