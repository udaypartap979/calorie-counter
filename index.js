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


console.log("META_PAGE_ACCESS_TOKEN length:", (process.env.META_PAGE_ACCESS_TOKEN || "").length);




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
You are a nutrition facts engine.

Priority order for nutrition sources:
1. Brand label / official restaurant menu (if available).
2. Restaurant menu approximations (if user mentions a venue like "Oberoi Mumbai" or "Punjab Grill"). 
   - Use estimates from known restaurant dishes or close analogs.
   - Clearly state assumptions (portion, restaurant estimate).
   - Confidence ≤0.6 unless official numbers are found.
3. Trusted DBs (USDA, IFCT, Nutritionix, Open Food Facts).

STRICT RULES:
- Never return 0 calories if the item is clearly edible. Always provide best-effort estimates with assumptions.
- Use "source": "venue_menu:<venue>" when you are giving a venue-based approximation.
- Always include "assumptions" array at item level and totals.
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
You are a vision nutrition parser.

Priority order for nutrition sources:
1. Brand label / official restaurant menu (if visible or mentioned in caption).
2. Restaurant menu approximations (if caption includes a venue like "Oberoi Mumbai" or "Punjab Grill").
   - Use estimates from known restaurant dishes or close analogs.
   - Clearly state assumptions (portion size, restaurant approximation).
   - Confidence ≤0.6 unless official numbers are found.
3. Trusted DBs (USDA, IFCT, Nutritionix, Open Food Facts).

STRICT RULES:
- Never return 0 calories if the image clearly shows edible food.
- For each food item, ALWAYS include an "assumptions" array at the item level.
- Totals must also include an "assumptions" array.
- Use "source": "venue_menu:<venue>" if the nutrition is an approximation from a restaurant.
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
// modify signature and use caption
async function classifyFoodOrWorkoutFromImage(imageBase64, mimeType, captionText = "") {
  try {
    const messages = [
      { role: "system", content: "Look at the image and caption and return only one word: 'food' or 'workout'." },
      { role: "user", content: [
          { type: "text", text: captionText ? `User caption (authoritative):\n"""${captionText}"""` : "No caption provided." },
          { type: "image_url", image_url: { url: `data:${mimeType};base64,${imageBase64}` } }
      ] }
    ];
    const r = await openai.chat.completions.create({ model: OPENAI_MODEL_VISION, temperature: 0.0, messages });
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
/*
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
*/
// Deterministic transcription: convert to 16k mono PCM WAV then call Whisper with language hint.
// -- Whisper transcription (with preprocessing re-enabled) --
async function transcribeAudioWithWhisper(audioBuffer) {
  const stamp = Date.now();
  const rawPath = `/tmp/${stamp}-raw.audio`;     // original format
  const cleanedPath = `/tmp/${stamp}-cleaned.wav`; // preprocessed PCM wav

  fs.writeFileSync(rawPath, audioBuffer);

  let stream = null;
  try {
    // Preprocess (rnnoise + silence removal + loudnorm if rnnoise model available)
    const pathForWhisper = await preprocessAudio(rawPath, cleanedPath);

    if (!fs.existsSync(pathForWhisper)) {
      throw new Error(`Audio file missing before Whisper: ${pathForWhisper}`);
    }

    stream = fs.createReadStream(pathForWhisper);

    const resp = await openai.audio.transcriptions.create({
      file: stream,
      model: "whisper-1",
      prompt: WHISPER_CONTEXT_PROMPT || "",
      language: "en"
      // language: "en",   // uncomment if you want to force English only
    });

    console.log("[Whisper] transcription OK:", resp?.text?.slice(0, 200));
    return resp.text ?? "";
  } catch (err) {
    console.error("Error transcribing with Whisper (preprocess path):", err);
    throw new Error("Failed to transcribe audio.");
  } finally {
    try { if (stream) stream.close(); } catch {}
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
async function analyzeAudioWithGPT(audioBuffer) {
  const transcript = await transcribeAudioWithWhisper(audioBuffer);
  console.log("[Audio->Transcript]", transcript);

  const prompt = `
Transcript: """${transcript}"""

Task:
- Extract food mentions (items, portions, nutrition) using the food parser schema.
- Extract workout mentions (activities, durations, calories burned) using the workout parser schema.
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

  // Primary call: ask for strict JSON
  const resp = await openai.chat.completions.create({
    model: OPENAI_MODEL_TEXT,
    temperature: 0.2,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: "You are a strict JSON generator for food/workout logs." },
      { role: "user", content: prompt }
    ]
  });

  let out = {};
  try {
    out = JSON.parse(resp.choices?.[0]?.message?.content || "{}");
  } catch (err) {
    console.error("[analyzeAudioWithGPT] JSON parse failed:", err);
    out = {
      food: { type: "food", details: [], totals: { calories: 0, assumptions: ["parse failed"], confidence: 0 } },
      workout: { type: "workout", details: [], totals: { calories_burned: 0, assumptions: ["parse failed"], confidence: 0 } },
      transcript
    };
  }

  // Defensive normalization
  out.food = out.food || { type: "food", details: [], totals: { calories: 0, assumptions: [], confidence: 0 } };
  out.workout = out.workout || { type: "workout", details: [], totals: { calories_burned: 0, assumptions: [], confidence: 0 } };

  // Defensive normalization for workout calories (if details include calories_burned)
  try {
    if (out.workout && Array.isArray(out.workout.details)) {
      const workoutTotal = out.workout.details.reduce(
        (s, d) => s + (num(d?.calories_burned, 0) || 0),
        0
      );

      if (workoutTotal > 0 && (!out.workout.totals || !out.workout.totals.calories_burned)) {
        out.workout.totals = out.workout.totals || {};
        out.workout.totals.calories_burned = workoutTotal;
        out.workout.totals.assumptions = Array.isArray(out.workout.totals.assumptions)
          ? out.workout.totals.assumptions.concat(["calculated from details"])
          : ["calculated from details"];
        out.workout.totals.confidence = out.workout.totals.confidence ?? 0.7;
        console.log(`[analyzeAudioWithGPT] recomputed workout calories=${workoutTotal}`);
      }
    }
  } catch (err) {
    console.error("[analyzeAudioWithGPT] workout normalization error:", err);
  }



  // If food items exist but calories total is zero, attempt targeted re-query per-item
  try {
    const details = Array.isArray(out.food.details) ? out.food.details : [];
    const totalCalories = details.reduce((s, d) => s + (num(d?.calories, 0) || 0), 0);

    if (details.length > 0 && totalCalories === 0) {
      console.log("[analyzeAudioWithGPT] Detected food items but calories == 0, attempting per-item nutrition lookup...");

      const enrichedDetails = [];

      for (const item of details) {
        const itemName = (item && item.item) ? String(item.item).trim() : "";
        if (!itemName) {
          enrichedDetails.push(item);
          continue;
        }

        // Prompt GPT specifically to extract nutrition for this single item using the strict food parser schema
        try {
          const foodResp = await openai.chat.completions.create({
            model: OPENAI_MODEL_TEXT,
            temperature: 0.2,
            response_format: { type: "json_object" },
            messages: [
              { role: "system", content: SYS_FOOD_TEXT },
              { role: "user", content: USER_FOOD_TEXT(itemName) }
            ],
          });

          let parsed = {};
          try {
            parsed = JSON.parse(foodResp.choices?.[0]?.message?.content || "{}");
          } catch (parseErr) {
            parsed = null;
          }

          if (parsed && parsed.type === "food" && Array.isArray(parsed.details) && parsed.details.length > 0) {
            // Prefer first detail returned for this item
            const pd = parsed.details[0];
            enrichedDetails.push({
              item: pd?.item ?? itemName,
              quantity: Number(pd?.quantity ?? item?.quantity ?? 0) || 0,
              unit: pd?.unit ?? item?.unit ?? "",
              calories: Number(pd?.calories ?? item?.calories ?? 0) || 0,
              macros: {
                protein: Number(pd?.macros?.protein ?? item?.macros?.protein ?? 0) || 0,
                fat:     Number(pd?.macros?.fat ?? item?.macros?.fat ?? 0) || 0,
                carbs:   Number(pd?.macros?.carbs ?? item?.macros?.carbs ?? 0) || 0,
              },
              brand: pd?.brand ?? item?.brand ?? "",
              source: pd?.source ?? item?.source ?? "",
              confidence: Math.max(0, Math.min(1, Number(pd?.confidence ?? item?.confidence ?? 0))) || 0,
              assumptions: Array.isArray(pd?.assumptions) ? pd.assumptions : (item?.assumptions || [])
            });
            continue;
          }

          // fallback: return original item unchanged if parser didn't help
          enrichedDetails.push(item);
        } catch (innerErr) {
          console.warn("[analyzeAudioWithGPT] per-item nutrition lookup failed for:", itemName, innerErr && (innerErr.message || innerErr));
          enrichedDetails.push(item);
        }
      } // end for

      // Replace details with enriched version and recompute totals
      out.food.details = enrichedDetails;
      const newTotal = enrichedDetails.reduce((s, d) => s + (num(d?.calories, 0) || 0), 0);

      out.food.totals = out.food.totals || {};
      out.food.totals.calories = newTotal;
      // keep assumptions array, but add a note that we enriched via per-item lookup
      out.food.totals.assumptions = Array.isArray(out.food.totals.assumptions) ? out.food.totals.assumptions.concat(["per-item nutrition lookup attempted"]) : ["per-item nutrition lookup attempted"];
      out.food.totals.confidence = out.food.totals.confidence ?? null;
      console.log(`[analyzeAudioWithGPT] per-item enrichment complete. new total calories=${newTotal}`);
    }
  } catch (enrichErr) {
    console.error("[analyzeAudioWithGPT] Error during per-item enrichment:", enrichErr);
    // don't fail the whole pipeline for enrichment errors
  }

  // --- Workout enrichment: if workout details exist but totals are zero, call workout estimator ---
  try {
    const wDetails = Array.isArray(out.workout.details) ? out.workout.details : [];
    const wTotalsVal = num(out.workout.totals?.calories_burned ?? 0, 0);

    if (wDetails.length > 0 && wTotalsVal === 0) {
      console.log("[analyzeAudioWithGPT] workout details present but calories_burned==0, calling workout estimator...");

      const wResp = await openai.chat.completions.create({
        model: OPENAI_MODEL_TEXT,
        temperature: 0.1,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: SYS_WORKOUT_ESTIMATOR },
          { role: "user", content: buildWorkoutUserPrompt({ modality: "text", text: transcript }) }
        ]
      });

      let wOut = {};
      try { wOut = JSON.parse(wResp.choices?.[0]?.message?.content || "{}"); } catch (e) { wOut = null; }

      if (wOut && wOut.type === "workout") {
        // Normalize details returned by GPT (ensure numeric duration_min and calories_burned)
        if (Array.isArray(wOut.details) && wOut.details.length) {
          out.workout.details = wOut.details.map(d => ({
            activity: (d?.activity || d?.name || "workout").toString(),
            duration_min: Number(d?.duration_min ?? d?.duration ?? 0) || 0,
            calories_burned: Number(d?.calories_burned ?? d?.calories ?? 0) || 0,
            intensity: d?.intensity || "unknown",
            assumptions: Array.isArray(d?.assumptions) ? d.assumptions : (d?.assumptions ? [d.assumptions] : []),
            confidence: (typeof d?.confidence === "number") ? d.confidence : (d?.confidence ? Number(d.confidence) : null)
          }));
        }

        // Ensure totals object exists and use GPT totals if provided (coerce to number)
        out.workout.totals = out.workout.totals || {};
        out.workout.totals.calories_burned = Number(wOut.totals?.calories_burned ?? out.workout.totals.calories_burned ?? 0) || 0;
        out.workout.totals.assumptions = Array.isArray(out.workout.totals.assumptions)
          ? out.workout.totals.assumptions.concat(wOut.totals?.assumptions || [])
          : (wOut.totals?.assumptions || []);
        out.workout.totals.confidence = out.workout.totals.confidence ?? (wOut.totals?.confidence ?? null);

        // If durations are all zero, attempt lightweight extraction from the transcript:
        try {
          const details = Array.isArray(out.workout.details) ? out.workout.details : [];
          const totalDur = details.reduce((s, d) => s + (Number(d.duration_min) || 0), 0);

          if (details.length && totalDur === 0 && typeof transcript === "string" && transcript.trim().length) {
            console.log("[analyzeAudioWithGPT] workout durations all zero — attempting to parse durations from transcript");

            // Extract numeric duration tokens (minutes/hours). This finds numbers like "30 min", "30 minutes", "1.5 hr", "1 hour"
            const minuteRegex = /(\d+(?:[\.,]\d+)?)\s*(?:m(?:in(?:utes?)?)?\.?)/gi;
            const hourRegex   = /(\d+(?:[\.,]\d+)?)\s*(?:h(?:our|rs?)?\.?)/gi;

            const foundDurations = [];

            let m;
            while ((m = minuteRegex.exec(transcript)) !== null) {
              const num = Number(String(m[1]).replace(",", "."));
              if (Number.isFinite(num)) foundDurations.push(Math.round(num)); // already minutes
            }
            while ((m = hourRegex.exec(transcript)) !== null) {
              const num = Number(String(m[1]).replace(",", "."));
              if (Number.isFinite(num)) foundDurations.push(Math.round(num * 60)); // convert hours -> minutes
            }

            // If nothing matched the strict patterns, also try looser "(\d+) ?mins" style matching
            if (!foundDurations.length) {
              const looser = /(\d{1,3})\s*(?:mins?|minutes?|m)/gi;
              while ((m = looser.exec(transcript)) !== null) {
                const num = Number(m[1]);
                if (Number.isFinite(num)) foundDurations.push(num);
              }
            }

            // If we found durations, assign them to workout details in order.
            if (foundDurations.length) {
              console.log("[analyzeAudioWithGPT] parsed durations (mins):", foundDurations);
              for (let i = 0; i < details.length; i++) {
                // round-robin or sequential assignment: prefer same-index duration, otherwise reuse last found
                const dur = foundDurations[i] ?? foundDurations[foundDurations.length - 1];
                details[i].duration_min = Number(dur) || details[i].duration_min || 0;
              }
              // Recalculate totals.calories_burned from detail-level calories if present (sum), else keep existing totals
              const sumCaloriesFromDetails = details.reduce((s, d) => s + (Number(d.calories_burned) || 0), 0);
              if (sumCaloriesFromDetails > 0) {
                out.workout.totals.calories_burned = Math.round(sumCaloriesFromDetails);
              }
              out.workout.details = details;
            } else {
              console.log("[analyzeAudioWithGPT] no durations parsed from transcript");
            }
          }
        } catch (durErr) {
          console.warn("[analyzeAudioWithGPT] duration extraction failed:", durErr && (durErr.message || durErr));
        }

        console.log("[analyzeAudioWithGPT] workout estimator provided calories_burned =", out.workout.totals.calories_burned, "details:", out.workout.details);
      }
    }
  } catch (err) {
    console.error("[analyzeAudioWithGPT] workout estimator error:", err);
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

// --- Meta-style webhook that mirrors your Twilio logic ---
// Requirements: process.env.META_PAGE_ACCESS_TOKEN, process.env.WHATSAPP_PHONE_NUMBER_ID
const FormDataNode = require("form-data");
// --- Paste near top of file ---
function prettyAxiosError(err) {
  try {
    if (!err) return String(err);
    if (err.response && err.response.data) {
      let d = err.response.data;
      // If it's an ArrayBuffer / Buffer -> string
      if (Buffer.isBuffer(d) || d instanceof ArrayBuffer) {
        try { d = Buffer.from(d).toString("utf8"); } catch {}
      } else if (typeof d === "object") {
        try { d = JSON.stringify(d, null, 2); } catch {}
      }
      return `HTTP ${err.response.status}: ${d}`;
    }
    if (err.request) {
      return `No response received. Request: ${err.request && err.request.path ? err.request.path : JSON.stringify(err.request)}`;
    }
    return `Error: ${err.message || String(err)}`;
  } catch (e) {
    return `Error pretty-printing axios error: ${String(e)}`;
  }
}

// --- safer meta send that prints readable errors ---
async function metaSendText(toWaId, text, contextMessageId = null) {
  const token = process.env.META_PAGE_ACCESS_TOKEN;
  const phoneNumberId = process.env.WHATSAPP_PHONE_NUMBER_ID;
  const graphVersion = process.env.META_GRAPH_VERSION || "v17.0";
  const payload = {
    messaging_product: "whatsapp",
    to: toWaId,
    type: "text",
    text: { body: text },
  };
  if (contextMessageId) payload.context = { message_id: contextMessageId };

  try {
    await axios.post(
      `https://graph.facebook.com/${graphVersion}/${phoneNumberId}/messages`,
      payload,
      { headers: { Authorization: `Bearer ${token}` } }
    );
  } catch (err) {
    console.error("[metaSendText] send error:", prettyAxiosError(err));
    throw err;
  }
}

// send an image message (by URL) via Meta/WhatsApp
async function metaSendImageUrl(toWaId, imageUrl, caption = "", contextMessageId = null) {
  const token = process.env.META_PAGE_ACCESS_TOKEN;
  const phoneNumberId = process.env.WHATSAPP_PHONE_NUMBER_ID;
  const graphVersion = process.env.META_GRAPH_VERSION || "v17.0";

  const payload = {
    messaging_product: "whatsapp",
    to: toWaId,
    type: "image",
    image: {
      link: imageUrl,
      caption: caption || undefined
    }
  };
  if (contextMessageId) payload.context = { message_id: contextMessageId };

  try {
    await axios.post(
      `https://graph.facebook.com/${graphVersion}/${phoneNumberId}/messages`,
      payload,
      { headers: { Authorization: `Bearer ${token}` } }
    );
  } catch (err) {
    console.error("[metaSendImageUrl] send error:", prettyAxiosError(err));
    throw err;
  }
}

async function fetchMetaMediaBytes(mediaId, opts = {}) {
  // opts: { maxRetries: number, retryDelayMs: number }
  const token = process.env.META_PAGE_ACCESS_TOKEN;
  const graphVersion = process.env.META_GRAPH_VERSION || "v17.0";
  const maxRetries = (opts.maxRetries != null) ? opts.maxRetries : 3;
  const baseDelay = (opts.retryDelayMs != null) ? opts.retryDelayMs : 300; // ms

  // 1) Fetch media object (gives temporary url)
  let metaResp;
  try {
    metaResp = await axios.get(
      `https://graph.facebook.com/${graphVersion}/${mediaId}`,
      { params: { fields: "url,mime_type", access_token: token }, timeout: 10_000 }
    );
  } catch (err) {
    console.error("[fetchMetaMediaBytes] error fetching media object:", prettyAxiosError(err));
    throw err;
  }

  const { url, mime_type } = metaResp.data || {};
  if (!url) throw new Error("Media object returned no 'url' field.");

  // Helper to try downloading (with optional auth header)
  async function tryDownload(downloadUrl, useAuth = false) {
    const headers = {};
    if (useAuth) headers.Authorization = `Bearer ${token}`;
    const resp = await axios.get(downloadUrl, { responseType: "arraybuffer", headers, timeout: 20000, validateStatus: s => true });
    return resp;
  }

  // 2) Try without auth first, then with auth if needed
  let attempt = 0;
  let lastError = null;
  while (attempt <= maxRetries) {
    const useAuth = attempt > 0; // 0 -> no auth, 1+ -> with auth (and exponential backoff)
    try {
      const resp = await tryDownload(url, useAuth);
      // Accept 200 as success; treat 2xx as success
      if (resp.status >= 200 && resp.status < 300) {
        return { buffer: Buffer.from(resp.data), mimeType: mime_type || "application/octet-stream" };
      }
      // If 401 and we didn't use auth yet, we'll retry with auth on next loop
      lastError = { status: resp.status, data: resp.data };
      const bodyPreview = Buffer.isBuffer(resp.data) ? resp.data.toString("utf8").slice(0, 300) : String(resp.data).slice(0,300);
      console.warn(`[fetchMetaMediaBytes] download attempt ${attempt} status=${resp.status} useAuth=${useAuth} body=${bodyPreview}`);
      // if status is 401 and we haven't tried auth, next attempt will set useAuth=true
    } catch (err) {
      lastError = err;
      console.warn(`[fetchMetaMediaBytes] download attempt ${attempt} failed: ${prettyAxiosError(err)}`);
    }

    // exponential backoff
    attempt++;
    const delay = baseDelay * Math.pow(2, attempt);
    await new Promise((r) => setTimeout(r, delay));
  }

  // all retries failed
  console.error("[fetchMetaMediaBytes] all download attempts failed. Last error:", lastError && (lastError.message || JSON.stringify(lastError)).slice ? (lastError.message || JSON.stringify(lastError)) : String(lastError));
  throw new Error("Failed to download media after retries.");
}



// local helper: logAnalysis (calls your existing /log-analysis endpoint)
async function logAnalysisMeta({ analysis, mediaBuffer, contentType, userPhoneNumber }) {
  try {
    const form = new FormDataNode();
    if (mediaBuffer) {
      // attempt to pick a sensible filename
      const ext = (contentType || "").includes("audio") ? ".ogg" : ".jpg";
      form.append("image", mediaBuffer, { filename: `log-media${ext}`, contentType: contentType || "application/octet-stream" });
    }
    form.append("userId", userPhoneNumber);
    form.append("userEmail", `${userPhoneNumber}@wa`); // or set to your user mapping
    form.append("analysisResult", JSON.stringify(analysis));
    await axios.post(`${BASE_URL}/log-analysis`, form, { headers: form.getHeaders() });
  } catch (err) {
    console.warn("logAnalysisMeta failed:", err?.response?.data || err.message);
  }
}
// ---------- Meta-style webhook (robust, keeps your original audio flow + improvements) ----------
app.post(
  "/whatsapp-webhook",
  express.json({ verify: (req, res, buf) => { req.rawBody = buf && buf.toString ? buf.toString() : null; } }),
  async (req, res) => {
    try {
      console.log("[INCOMING WEBHOOK BODY]", JSON.stringify(req.body || {}, null, 2));
      const incoming = req.body;
      if (!incoming || !incoming.entry) {
        console.log("[WH] no entry -> 200");
        return res.sendStatus(200);
      }

      // collect tasks
      const tasks = [];
      for (const entry of incoming.entry) {
        const changes = entry.changes || [];
        for (const change of changes) {
          const val = change.value || {};
          const messages = val.messages || [];
          for (const message of messages) {
            // debug
            console.log("----- [WH MSG DEBUG] BEGIN -----");
            console.log("[WH MSG] keys:", Object.keys(message || {}));
            try { console.log("[WH MSG] preview:", JSON.stringify(message, null, 2).slice(0, 4000)); } catch (e) {}
            console.log("[WH MSG] hasImage:", !!message?.image, "hasVideo:", !!message?.video, "hasDocument:", !!message?.document, "hasAudio:", !!message?.audio);
            console.log("----- [WH MSG DEBUG] END -----");

            const from = message.from;
            const msgId = message.id;
            tasks.push({ from, msgId, message, metadata: val.metadata || {} });
          }
        }
      }

      if (!tasks.length) {
        console.log("[WH] no tasks -> 200");
        return res.sendStatus(200);
      }

      // Quick ack to each sender (de-duplicated)
      const ackText = "Processing your request...";
      const acked = new Set();
      for (const t of tasks) {
        try {
          if (t.from && !acked.has(t.from)) {
            await metaSendText(t.from, ackText, t.msgId).catch(err => console.warn("[WH] ack send failed:", err && err.message ? err.message : err));
            acked.add(t.from);
          }
        } catch (e) {}
      }

      // respond to webhook quickly (Meta expects a fast response)
      res.status(200).send();

      // Process tasks sequentially (same-process)
      for (const t of tasks) {
        const { from, msgId, message } = t;
        try {
          // robust caption extraction
          const captionCandidates = [
            message?.text?.body,
            message?.caption,
            message?.image?.caption,
            message?.image?.caption?.text,
            message?.context?.quoted_message?.text
          ];
          let captionText = "";
          for (const p of captionCandidates) {
            if (p && typeof p === "string" && p.trim().length) {
              captionText = p.trim();
              break;
            }
          }
          if (captionText) {
            captionText = captionText.replace(/\r?\n+/g, " ").trim();
            if (captionText.length > 800) captionText = captionText.slice(0, 800) + "...";
          }

          // handle media messages
          if (message.image || message.video || message.document || message.audio) {
            const mediaObj = message.image || message.video || message.document || message.audio;
            const mediaId = mediaObj?.id;
            if (!mediaId) {
              await metaSendText(from, "Sorry, couldn't find media to download. Please resend.", msgId);
              continue;
            }

            // download media (handles auth / retries)
            let mediaBuffer, contentType;
            try {
              const fetched = await fetchMetaMediaBytes(mediaId);
              mediaBuffer = fetched.buffer;
              contentType = fetched.mimeType || fetched.mime_type || (mediaObj?.mime_type || "");
            } catch (err) {
              console.error("[whatsapp-webhook] failed to download media:", prettyAxiosError(err));
              await metaSendText(from, "Sorry, I couldn't download that media. Please try sending it again.", msgId);
              continue;
            }

            // IMAGE handling
            if ((contentType || "").startsWith("image/") || message.image) {
              try {
                // If you want to call your internal merged analyzer directly instead of HTTP,
                // you can call analyzeImageWithChatGPT(mediaBuffer, contentType, captionText)
                // But to preserve the existing flow that merges caption-first + vision, call /analyze-image-with-text as before:
                const form = new FormDataNode();
                const filename = "whatsapp-image.jpg";
                form.append("image", mediaBuffer, { filename, contentType: contentType || "image/jpeg" });
                if (captionText) form.append("text", captionText);

                const safeCaptionForUrl = captionText ? `?text=${encodeURIComponent(captionText)}` : "";
                const analyzeUrl = `${BASE_URL}/analyze-image-with-text${safeCaptionForUrl}`;
                console.log("[WH -> ANALYZE] calling analyze endpoint:", analyzeUrl);
                console.log("[WH -> ANALYZE] form headers preview:", form.getHeaders()['content-type'] ? form.getHeaders()['content-type'].slice(0,200) : "<no content-type>");

                const { data } = await axios.post(analyzeUrl, form, { headers: form.getHeaders(), timeout: 60000 });
                const analysis = data;
                // log & send reply
                await logAnalysisMeta({ analysis, mediaBuffer, contentType, userPhoneNumber: from }).catch(e => console.warn("[WH] logAnalysisMeta failed:", e && e.message));
                const reply = buildReplyForAnalysis(analysis);
                await metaSendText(from, reply, msgId);
              } catch (err) {
                console.error("[WH -> ANALYZE] analyze call failed:", prettyAxiosError(err));
                await metaSendText(from, "Sorry — failed to analyze the image. Try again.", msgId);
              }
              continue;
            }

            // AUDIO handling (voice note)
            if ((contentType || "").startsWith("audio/") || (contentType || "").includes("ogg") || mediaObj?.voice) {
              try {
                console.log("[whatsapp-webhook][audio] processing audio for", from);
                // debug write
                const stamp = Date.now();
                const ext = (contentType || "").includes("ogg") ? ".ogg" : (contentType || "").includes("mpeg") || (contentType || "").includes("mp3") ? ".mp3" : ".audio";
                const debugPath = `/tmp/wa-voice-${stamp}${ext}`;
                try { fs.writeFileSync(debugPath, mediaBuffer); console.log("[whatsapp-webhook][audio] saved debug file:", debugPath, "bytes:", mediaBuffer.length); } catch (e) { console.warn("[whatsapp-webhook][audio] failed to write debug file:", e && e.message); }

                // Quick ffmpeg check (useful if your preprocess uses ffmpeg)
                try {
                  const { stdout } = await execPromise("ffmpeg -version").catch(() => ({ stdout: "" }));
                  if (!stdout) console.warn("[whatsapp-webhook][audio] ffmpeg not found; ensure ffmpeg installed for preprocessing/transcription.");
                  else console.log("[whatsapp-webhook][audio] ffmpeg available.");
                } catch (e) { console.warn("[whatsapp-webhook][audio] ffmpeg check error:", e && e.message); }

                // 1) Primary: in-process analyze (transcribe -> analyze)
                let combinedAnalysis = null;
                try {
                  combinedAnalysis = await analyzeAudioWithGPT(mediaBuffer);
                  console.log("[whatsapp-webhook][audio] analyzeAudioWithGPT succeeded");
                } catch (errPrimary) {
                  console.error("[whatsapp-webhook][audio] analyzeAudioWithGPT failed:", prettyAxiosError(errPrimary));
                }

                // 2) Fallback: processAndLogAudio (multipart POST to /analyze-audio + /log-analysis)
                if (!combinedAnalysis) {
                  try {
                    console.log("[whatsapp-webhook][audio] attempting fallback processAndLogAudio...");
                    combinedAnalysis = await processAndLogAudio(mediaBuffer, from, contentType || "audio/ogg", "whatsapp-audio");
                    console.log("[whatsapp-webhook][audio] processAndLogAudio succeeded");
                  } catch (errFallback) {
                    console.error("[whatsapp-webhook][audio] processAndLogAudio failed:", prettyAxiosError(errFallback));
                  }
                }

                // 3) If both failed
                if (!combinedAnalysis) {
                  console.error("[whatsapp-webhook][audio] both primary and fallback analyzers failed. Debug file:", debugPath);
                  await metaSendText(from, `Sorry — I couldn't process your voice note right now. I've saved the file for debugging: ${debugPath}\nPlease try a short (3–8s) voice note or send the text.`, msgId);
                  continue;
                }

                // 4) Build reply
                let reply = "";
                // If analyzer returned the { food, workout, transcript } shape (analyzeAudioWithGPT)
                if (combinedAnalysis && (combinedAnalysis.food || combinedAnalysis.workout)) {
                  const hasFood = combinedAnalysis.food && Array.isArray(combinedAnalysis.food.details) && combinedAnalysis.food.details.length;
                  const hasWorkout = combinedAnalysis.workout && Array.isArray(combinedAnalysis.workout.details) && combinedAnalysis.workout.details.length;

                  if (hasFood && !hasWorkout) {
                    reply = buildReplyForAnalysis(combinedAnalysis.food);
                  } else if (hasWorkout && !hasFood) {
                    reply = buildReplyForAnalysis(combinedAnalysis.workout);
                  } else if (hasFood && hasWorkout) {
                    const f = buildReplyForAnalysis(combinedAnalysis.food);
                    const w = buildReplyForAnalysis(combinedAnalysis.workout);
                    reply = `${f}\n\n${w}`;
                  } else {
                    const tr = combinedAnalysis.transcript ? `Transcript: "${String(combinedAnalysis.transcript).slice(0,240)}"` : "";
                    reply = `Sorry, I couldn't confidently extract food or workout from the voice note. ${tr}\nTry a short 3–8s voice note or type it.`;
                  }
                }
                // If fallback returned analysis shaped like your /analyze-audio or /analyze-image result
                else if (combinedAnalysis && combinedAnalysis.type) {
                  reply = buildReplyForAnalysis(combinedAnalysis);
                } else {
                  reply = "Sorry — analysis returned an unexpected format. Try text or a short voice note.";
                }

                // 5) Ensure logged (if fallback hasn't already logged)
                try { await logAnalysisMeta({ analysis: combinedAnalysis, mediaBuffer, contentType, userPhoneNumber: from }); } catch (e) { console.warn("[whatsapp-webhook][audio] logAnalysisMeta non-fatal error:", e && e.message); }

                // 6) send reply
                await metaSendText(from, reply, msgId);
              } catch (audioErr) {
                console.error("[whatsapp-webhook][audio] unexpected error:", audioErr && (audioErr.stack || audioErr.message || audioErr));
                try { await metaSendText(from, "Sorry — couldn't process your voice note right now. Try sending a short one or text.", msgId); } catch {}
              }
              continue;
            }

            // If media type not recognized:
            await metaSendText(from, "Sorry, I can process photos and voice notes only. Please send a photo or a short voice note.", msgId);
            continue;
          } // end media handling

          // handle text messages
          else if (message.text && message.text.body) {
            const userText = message.text.body;
            try {
              const lower = userText.toLowerCase();
              if (lower.includes("crave") || lower.includes("craving")) {
                const { data } = await axios.post(`${BASE_URL}/handle-craving`, { text: userText, userId: from });
                await metaSendText(from, data.advice, msgId);
              } else if (lower.includes("what should i eat") || lower.includes("diet suggestion") || lower.includes("recommend food")) {
                const { data } = await axios.post(`${BASE_URL}/recommend-food`, { userId: from });
                await metaSendText(from, data.recommendations, msgId);
              } else if (lower.includes("analyze") || lower.includes("summary")) {
                const { data } = await axios.post(`${BASE_URL}/analyze-summary`, { userId: from });
                try {
                  await metaSendImageUrl(from, data.caloriesChartUrl, "📊 Here’s your Calories Summary", msgId);
                  await metaSendImageUrl(from, data.macrosChartUrl, "📊 Here’s your Macros Summary", msgId);
                } catch (sendErr) {
                  console.warn("Failed to send chart images:", sendErr?.response?.data || sendErr?.message || sendErr);
                  await metaSendText(from, `Charts:\n${data.caloriesChartUrl}\n${data.macrosChartUrl}`, msgId);
                }
              } else {
                // default: analyze text (food/workout)
                try {
                  const { data: analysis } = await axios.post(`${BASE_URL}/analyze-text`, { text: userText });
                  await logAnalysisMeta({ analysis, mediaBuffer: null, contentType: null, userPhoneNumber: from });
                  const reply = buildReplyForAnalysis(analysis);
                  await metaSendText(from, reply, msgId);
                } catch (err) {
                  console.error("[whatsapp-webhook][text] error:", prettyAxiosError(err));
                  await metaSendText(from, "Sorry — couldn't analyze that text right now.", msgId);
                }
              }
            } catch (err) {
              console.error("[whatsapp-webhook][text] unexpected error:", err && (err.stack || err.message || err));
              try { await metaSendText(from, "Sorry — couldn't process that message.", msgId); } catch {}
            }
          } // interactive (buttons/lists)
          else if (message.interactive) {
            const title = message.interactive.button_reply?.title || message.interactive.list_reply?.title || "";
            if (title) {
              const { data: analysis } = await axios.post(`${BASE_URL}/analyze-text`, { text: title });
              await logAnalysisMeta({ analysis, mediaBuffer: null, contentType: null, userPhoneNumber: from });
              const reply = buildReplyForAnalysis(analysis);
              await metaSendText(from, reply, msgId);
            } else {
              await metaSendText(from, "Thanks — I received your selection.", msgId);
            }
          } else {
            await metaSendText(from, "Sorry, I can process text, photos and voice notes. Please send one of those.", msgId);
          }
        } catch (procErr) {
          console.error("Error processing Meta WhatsApp message:", prettyAxiosError(procErr));
          try { await metaSendText(t.from, "Food/Workout: Sorry, I couldn't process that. Please try again.", t.msgId); } catch {}
        }
      } // end tasks loop
    } catch (outerErr) {
      console.error("[whatsapp-webhook] outer handler error:", prettyAxiosError(outerErr));
      try { res.status(500).send(); } catch {}
    }
  }
);




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

// Add this near your other classifier helpers (e.g. under classifyFoodOrWorkoutFromImage)
async function classifyFoodOrWorkoutFromText(text) {
  try {
    // Defensive: coerce to string
    const input = (text || "").toString().trim();
    if (!input) return "food";

    const resp = await openai.chat.completions.create({
      model: OPENAI_MODEL_TEXT,
      temperature: 0.0,
      messages: [
        { role: "system", content: "Classify the input as exactly one word: 'food' or 'workout'. Return only that word." },
        { role: "user", content: `INPUT:\n"""${input}"""` }
      ],
      // you can keep response_format off here since we only need plain text
    });

    const content = String(resp?.choices?.[0]?.message?.content || "").toLowerCase();
    if (content.includes("workout")) return "workout";
    return "food";
  } catch (err) {
    console.warn("[classifyFoodOrWorkoutFromText] error:", err && (err.message || err));
    // safe default: treat as food so text food flows still work
    return "food";
  }
}


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
// ---------- ANALYZE IMAGE + TEXT (merge caption-only items + vision-detected items) ----------
app.post("/analyze-image-with-text", upload.single("image"), async (req, res) => {
  try {
    // Basic request debug
    console.log("[/analyze-image-with-text] content-type:", req.headers['content-type']);
    console.log("[/analyze-image-with-text] req.body keys:", Object.keys(req.body || {}));
    console.log("[/analyze-image-with-text] req.query keys:", Object.keys(req.query || {}));
    console.log("[/analyze-image-with-text] req.file present:", !!req.file);
    if (req.file) console.log("[/analyze-image-with-text] file:", { name: req.file.originalname, mimetype: req.file.mimetype, size: req.file.size });

    // 1) Collect caption (body multipart or query fallback)
    const captionFromBody = req.body && req.body.text ? String(req.body.text).trim() : "";
    const captionFromQuery = req.query && req.query.text ? String(req.query.text).trim() : "";
    const caption = captionFromBody || captionFromQuery || "";

    console.log("[/analyze-image-with-text] caption present?:", !!caption);
    if (caption) console.log("[/analyze-image-with-text] caption preview:", caption.slice(0,400));

    // Helper: safe JSON parse
    const safeParse = (s) => { try { return JSON.parse(s || "{}"); } catch (e) { return null; } };

    // Storage for parsed outputs
    let captionParsed = null; // result from text parser (food/workout schema)
    let visionParsed = null;  // result from vision model

    // 2) If caption exists: parse it strictly (food/workout). We will use these items as authoritative for names present.
    if (caption) {
      try {
        // 2a: classify caption as food or workout
        let cls = "food";
        try {
          const clsResp = await openai.chat.completions.create({
            model: OPENAI_MODEL_TEXT,
            temperature: 0.0,
            messages: [
              { role: "system", content: "Classify the input as 'food' or 'workout'. Return only one word." },
              { role: "user", content: `INPUT:\n"""${caption}"""` }
            ]
          });
          const clsText = String(clsResp.choices?.[0]?.message?.content || "").toLowerCase();
          cls = clsText.includes("workout") ? "workout" : "food";
        } catch (e) {
          console.warn("[/analyze-image-with-text] caption classification failed, defaulting to food:", e && e.message);
          cls = "food";
        }

        if (cls === "workout") {
          // Call workout estimator
          const wResp = await openai.chat.completions.create({
            model: OPENAI_MODEL_TEXT,
            temperature: 0.1,
            messages: [
              { role: "system", content: SYS_WORKOUT_ESTIMATOR },
              { role: "user", content: buildWorkoutUserPrompt({ modality: "text", text: caption }) }
            ]
          });

          const wOut = safeParse(wResp.choices?.[0]?.message?.content) || null;
          if (wOut && wOut.type === "workout") captionParsed = wOut;
        } else {
          // Call strict food parser
          const fResp = await openai.chat.completions.create({
            model: OPENAI_MODEL_TEXT,
            temperature: 0.0,
            messages: [
              { role: "system", content: SYS_FOOD_TEXT },
              { role: "user", content: USER_FOOD_TEXT(caption) }
            ]
          });

          const fOut = safeParse(fResp.choices?.[0]?.message?.content) || null;
          if (fOut && fOut.type === "food") {
            // annotate items as user_caption and bump confidence
            if (Array.isArray(fOut.details)) {
              fOut.details = fOut.details.map(d => ({
                item: d?.item ?? "",
                quantity: Number(d?.quantity ?? 0) || 0,
                unit: d?.unit ?? "",
                calories: Number(d?.calories ?? 0) || 0,
                macros: {
                  protein: Number(d?.macros?.protein ?? 0) || 0,
                  fat: Number(d?.macros?.fat ?? 0) || 0,
                  carbs: Number(d?.macros?.carbs ?? 0) || 0
                },
                brand: d?.brand ?? "",
                source: "user_caption",
                confidence: Math.max(0.9, Number(d?.confidence ?? 0.9)),
                assumptions: Array.isArray(d?.assumptions) ? d.assumptions : (d?.assumptions ? [String(d.assumptions)] : ["parsed from caption"])
              }));
            }
            fOut.totals = fOut.totals || {};
            fOut.totals.calories = Number(fOut.totals?.calories ?? fOut.details.reduce((s, it) => s + num(it.calories, 0), 0)) || 0;
            fOut.totals.assumptions = (fOut.totals.assumptions || []).concat(["caption-first parse"]);
            fOut.totals.confidence = fOut.totals.confidence ?? 0.95;
            captionParsed = fOut;
          }
        }
      } catch (captionErr) {
        console.warn("[/analyze-image-with-text] caption parse error - falling back to vision:", captionErr && (captionErr.message || captionErr));
        captionParsed = null;
      }
    } // end caption parse

    // 3) Always run the vision parser on the image (so we detect items not in caption)
    if (!req.file) {
      // If no file and captionParsed exists, just return captionParsed (already handled above), else error
      if (captionParsed) return res.status(200).json(captionParsed);
      return res.status(400).json({ error: "No image file provided." });
    }

    // vision call
    const imageBase64 = req.file.buffer.toString("base64");
    const mimeType = req.file.mimetype || "image/jpeg";
    const userTextBlock = caption ? `${USER_FOOD_IMAGE}\n\nExtra context from user caption: ${caption}` : USER_FOOD_IMAGE;

    try {
      const vResp = await openai.chat.completions.create({
        model: OPENAI_MODEL_VISION,
        temperature: 0.0,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: SYS_FOOD_IMAGE },
          {
            role: "user",
            content: [
              { type: "text", text: userTextBlock },
              { type: "image_url", image_url: { url: `data:${mimeType};base64,${imageBase64}` } }
            ]
          }
        ]
      });

      visionParsed = safeParse(vResp.choices?.[0]?.message?.content) || null;
      if (!visionParsed || visionParsed.type !== "food") {
        visionParsed = { type: "food", details: [], totals: { calories: 0, assumptions: [], confidence: 0 } };
      }
    } catch (visionErr) {
      console.warn("[/analyze-image-with-text] vision parse failed, continuing with caption if any:", visionErr && (visionErr.message || visionErr));
      visionParsed = { type: "food", details: [], totals: { calories: 0, assumptions: [], confidence: 0 } };
    }

    // Defensive normalize vision details
    visionParsed.details = Array.isArray(visionParsed.details) ? visionParsed.details.map(d => ({
      item: (d?.item ?? "").toString(),
      quantity: Number(d?.quantity ?? 0) || 0,
      unit: d?.unit ?? "",
      calories: Number(d?.calories ?? 0) || 0,
      macros: {
        protein: Number(d?.macros?.protein ?? 0) || 0,
        fat: Number(d?.macros?.fat ?? 0) || 0,
        carbs: Number(d?.macros?.carbs ?? 0) || 0
      },
      brand: d?.brand ?? "",
      source: d?.source ?? "vision",
      confidence: Math.max(0, Math.min(1, Number(d?.confidence ?? 0))) || 0,
      assumptions: Array.isArray(d?.assumptions) ? d.assumptions : (d?.assumptions ? [String(d.assumptions)] : [])
    })) : [];

    // 4) Merge captionParsed (preferred) + visionParsed (add unique items)
    const merged = [];
    const seen = new Set();

    // Helper to normalize a name for fuzzy matching
    const normalizeName = (n) => (n || "").toString().toLowerCase().replace(/[^a-z0-9\s]/g, "").trim();

    // Add caption items first (if any)
    if (captionParsed && Array.isArray(captionParsed.details) && captionParsed.details.length) {
      for (const c of captionParsed.details) {
        const n = normalizeName(c.item);
        if (!n) continue;
        merged.push(c);
        seen.add(n);
      }
    }

    // For each vision item, include only if not covered by caption (fuzzy include)
    for (const v of visionParsed.details) {
      const vn = normalizeName(v.item);
      if (!vn) continue;

      // check for exact or partial overlap with any seen caption names
      let conflict = false;
      for (const s of Array.from(seen)) {
        // if caption name contains vision or vision contains caption (covers "smoothie" vs "banana smoothie")
        if (s.includes(vn) || vn.includes(s)) {
          conflict = true;
          break;
        }
        // allow a loose token overlap check: share at least one word
        const sTokens = s.split(/\s+/).filter(Boolean);
        const vTokens = vn.split(/\s+/).filter(Boolean);
        if (sTokens.some(tok => vTokens.includes(tok))) {
          conflict = true;
          break;
        }
      }
      if (!conflict) {
        merged.push(v);
        seen.add(vn);
      } else {
        // If conflict, optionally attach vision assumptions to the caption item (find and merge)
        for (let i = 0; i < merged.length; i++) {
          const mName = normalizeName(merged[i].item);
          if (mName && (mName.includes(vn) || vn.includes(mName) || mName.split(/\s+/).some(tok => vn.split(/\s+/).includes(tok)))) {
            // append vision assumptions if not duplicate
            if (Array.isArray(v.assumptions) && v.assumptions.length) {
              merged[i].assumptions = Array.from(new Set((merged[i].assumptions || []).concat(v.assumptions)));
            }
            // optionally, keep a small note about vision confidence
            merged[i].assumptions = (merged[i].assumptions || []).concat([`vision_conf:${v.confidence || 0}`]);
            break;
          }
        }
      }
    }

    // 5) Build totals from merged details (sum calories) and set overall confidence as average
    let totalCalories = 0;
    const confs = [];
    for (const it of merged) {
      totalCalories += Number(it.calories || 0);
      if (typeof it.confidence === "number") confs.push(it.confidence);
      else if (!isNaN(Number(it.confidence))) confs.push(Number(it.confidence));
    }
    const overallConfidence = confs.length ? +(confs.reduce((a,b) => a+b, 0)/confs.length).toFixed(3) : null;

    const result = {
      type: "food",
      details: merged,
      totals: {
        calories: Math.round(totalCalories),
        assumptions: (captionParsed?.totals?.assumptions || []).concat(visionParsed?.totals?.assumptions || []).filter(Boolean),
        confidence: overallConfidence
      }
    };

    // Defensive: if nothing found, return visionParsed as fallback
    if (!result.details.length) {
      return res.status(200).json(visionParsed);
    }

    console.log("[/analyze-image-with-text] merged items:", result.details.map(d => d.item));
    return res.status(200).json(result);
  } catch (err) {
    console.error("Error in /analyze-image-with-text (merge):", err && (err.stack || err.message || err));
    return res.status(500).json({ error: "Failed to analyze image with text." });
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

