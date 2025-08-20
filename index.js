require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const OpenAI = require("openai"); // Added for new functions
const fs = require("fs"); // Added for new functions
const { GoogleGenerativeAI } = require("@google/generative-ai");
const nodemailer = require("nodemailer");
const supabase = require("./supabaseClient");
const twilio = require("twilio");

const app = express();
app.use(cors());
app.use(express.json());

const port = process.env.PORT || 3000;

// +++ ADDED: OpenAI Client for new functions +++
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// --- Existing Gemini, Multer, and Nodemailer Setup (Unchanged) ---
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
});
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "gurjeetchem@gmail.com",
    pass: "vsvb ltyz eqfp wleu",
  },
});

// --- Existing Spoonacular Functions (Unchanged) ---
/**
 * @param {string} foodName
 * @returns {Promise<object|null>} The first search result or null.
 */
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

/**
 * Gets detailed nutrition information for a specific ingredient ID.
 * @param {number} ingredientId - The Spoonacular ID for the ingredient.
 * @param {number} amount - The amount of the ingredient.
 * @param {string} unit - The unit for the amount (e.g., "grams").
 * @returns {Promise<object|null>} Detailed nutrition object or null.
 */
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

/**
 * Searches for a recipe and its nutrition data.
 * @param {string} dishName - The name of the dish.
 * @returns {Promise<object|null>} Recipe nutrition object or null.
 */
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

/**
 * A comprehensive function to get food info, trying ingredients first, then recipes.
 * @param {string} foodName - The name of the food.
 * @returns {Promise<object|null>} The best available nutrition info or null.
 */
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

/**
 * @param {string} foodName - The name of the food.
 * @returns {Promise<object|null>} A quick nutrition guess or null.
 */
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

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++ SECTION: NEW FUNCTIONS FOR CHATGPT AND WHISPER  +++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++

/**
 * Transcribes audio using OpenAI's Whisper API.
 * @param {Buffer} audioBuffer - The audio file buffer.
 * @returns {Promise<string>} The transcribed text.
 */
async function transcribeAudioWithWhisper(audioBuffer) {
  const tempFilePath = `/tmp/${Date.now()}.mp3`;
  fs.writeFileSync(tempFilePath, audioBuffer);
  try {
    const response = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempFilePath),
      model: "whisper-1",
    });
    return response.text;
  } catch (error) {
    console.error("Error transcribing with Whisper:", error);
    throw new Error("Failed to transcribe audio.");
  } finally {
    fs.unlinkSync(tempFilePath);
  }
}

/**
 * Analyzes a transcript with ChatGPT to classify and extract details.
 * @param {string} transcript - The text from the audio.
 * @returns {Promise<object>} The structured analysis from ChatGPT.
 */
async function analyzeTranscriptWithChatGPT(transcript) {
  const prompt = `
    Analyze the following transcript and first classify it as either "food" or "workout".

- If it is "food":
  • Identify each food item.
  • Estimate calories and macros (protein, fat, carbs in grams).
  • Provide a credible source (e.g., "USDA FoodData Central", "Nutritionix API", "General nutritional database").

- If it is "workout":
  • Identify each exercise.
  • Estimate calories burned using MET-based formulas adjusted to align with Apple Watch outputs.
  • Formula: Calories/min = (MET × 3.5 × body weight in kg ÷ 200).
  • Add an adjustment factor of +25% to approximate Apple Watch’s calorie reporting.
  • Assume body weight = 70 kg if not provided.

Respond ONLY with a JSON object. Do not include any other text or explanations.

Example for food:
{
  "type": "food",
  "details": [
    {
      "item": "Gobhi Parantha",
      "calories": 220,
      "macros": { "protein": 6, "fat": 10, "carbs": 28 },
      "source": "General nutritional database for Indian cuisine"
    }
  ]
}

Example for workout:
{
  "type": "workout",
  "details": [
    { "exercise": "30-minute run", "calories_burned": 420 }
  ]
}

    Transcript: "${transcript}"
  `;
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4-turbo",
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
    });
    return JSON.parse(response.choices[0].message.content);
  } catch (error) {
    console.error("Error analyzing transcript with ChatGPT:", error);
    throw new Error("Failed to analyze transcript.");
  }
}

/**
 * Analyzes text or image content with ChatGPT for nutrition.
 * @param {string} content - The text to analyze (e.g., "2 rotis and dal").
 * @returns {Promise<object>} The structured analysis from ChatGPT.
 */
async function analyzeContentWithChatGPT(content) {
  const prompt = `
    Analyze the following food description and provide a nutritional breakdown.
    - Identify each food item.
    - For each item, estimate its calories and macros (protein, fat, carbs in grams).
    - Provide a credible source for the nutritional data for each item.
    - Respond ONLY with a JSON object in the specified format. Do not add any extra text.

    Example format:
    {
      "details": [
        {
          "item": "Roti",
          "quantity": 2,
          "calories": 160,
          "macros": { "protein": 6, "fat": 4, "carbs": 30 },
          "source": "USDA FoodData Central"
        }
      ]
    }

    Content: "${content}"
  `;
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4-turbo",
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
    });
    return JSON.parse(response.choices[0].message.content);
  } catch (error) {
    console.error("Error analyzing content with ChatGPT:", error);
    throw new Error("Failed to analyze content.");
  }
}

// --- Existing Endpoints (Unchanged) ---
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

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++ SECTION: NEW ENDPOINTS FOR DIRECT AI ANALYSIS +++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++

/**
 * NEW ENDPOINT: Analyzes a voice note for food or workout details.
 */
app.post("/analyze-audio", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file provided." });
    }
    const transcript = await transcribeAudioWithWhisper(req.file.buffer);
    const analysis = await analyzeTranscriptWithChatGPT(transcript);
    res.status(200).json(analysis);
  } catch (error) {
    console.error("Error in /analyze-audio endpoint:", error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * NEW ENDPOINT: Analyzes a text description of a meal.
 */
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

/**
 * NEW ENDPOINT: Analyzes an image of a meal.
 */
app.post("/analyze-image", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file provided." });
    }
    
    // A single call to the new function
    const analysis = await analyzeImageWithChatGPT(req.file.buffer, req.file.mimetype);

    res.status(200).json(analysis);
  } catch (error) {
    console.error("Error in /analyze-image endpoint:", error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * Analyzes an image directly with ChatGPT (GPT-4o) for a full nutritional breakdown.
 * @param {Buffer} imageBuffer - The image file buffer.
 * @param {string} mimeType - The mime type of the image.
 * @returns {Promise<object>} The structured analysis from ChatGPT.
 */
async function analyzeImageWithChatGPT(imageBuffer, mimeType) {
  const imageBase64 = imageBuffer.toString("base64");

  const prompt = `
    Analyze the food in this image and provide a full nutritional breakdown.
    - First, identify each distinct food item.
    - For each item, estimate its calories and macros (protein, fat, carbs in grams).
    - Provide a credible source for the nutritional data for each item.
    - Respond ONLY with a JSON object in the specified format. Do not add any extra text or explanations.

    Example format:
    {
      "details": [
        {
          "item": "Dal Tadka",
          "quantity": 1,
          "calories": 180,
          "macros": { "protein": 8, "fat": 6, "carbs": 25 },
          "source": "General nutritional database for Indian cuisine"
        },
        {
          "item": "Roti",
          "quantity": 2,
          "calories": 200,
          "macros": { "protein": 6, "fat": 5, "carbs": 30 },
          "source": "Standard nutritional data for whole wheat flatbread"
        }
      ]
    }
  `;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o", // This is OpenAI's latest, powerful vision model
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image_url",
              image_url: {
                url: `data:${mimeType};base64,${imageBase64}`,
              },
            },
          ],
        },
      ],
      response_format: { type: "json_object" },
    });
    return JSON.parse(response.choices[0].message.content);
  } catch (error) {
    console.error("Error analyzing image with ChatGPT:", error);
    throw new Error("Failed to analyze image with ChatGPT.");
  }
}

app.post("/log-analysis", upload.single("image"), async (req, res) => {
  try {
    // analysisResult is sent as a JSON string in form data
    const { analysisResult, userId, userEmail } = req.body;

    if (!analysisResult || !userId || !userEmail) {
      return res.status(400).json({
        error: "analysisResult, userId, and userEmail are required.",
      });
    }

    const parsedAnalysis = JSON.parse(analysisResult);
    const { type = "food", details } = parsedAnalysis;
    let imageUrl = null;

    // Check if an image file was uploaded with the request
    if (req.file) {
      const fileName = `${Date.now()}-${req.file.originalname}`;
      // Upload the image buffer to Supabase Storage
      await supabase.storage
        .from("meal-images")
        .upload(fileName, req.file.buffer, { contentType: req.file.mimetype });
      
      // Get the public URL of the uploaded image
      const { data: urlData } = supabase.storage
        .from("meal-images")
        .getPublicUrl(fileName);
      imageUrl = urlData.publicUrl;
    }

    const totalCalories = details.reduce((sum, item) => {
      return sum + (Number(item.calories) || Number(item.calories_burned) || 0);
    }, 0);

    const newLog = {
      user_id: userId,
      user_email: userEmail,
      item_type: type,
      total_calories: Math.round(totalCalories),
      log_details: details,
      image_url: imageUrl, // Use the new URL, or null if no image was sent
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

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++ SECTION: NEW WHATSAPP BOT LOGIC                 +++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++

const twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
const BASE_URL = process.env.BASE_URL || `http://localhost:${port}`; // Your server's public URL

/**
 * Helper function to download media from Twilio's URL
 */
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

/**
 * NEW ENDPOINT: Receives incoming messages from WhatsApp via Twilio
 */
app.post("/whatsapp-webhook", async (req, res) => {
  const incomingMsg = req.body;
  const userPhoneNumber = incomingMsg.From; // e.g., 'whatsapp:+14155238886'
  let responseMessage = "Processing your request...";

  // Immediately send a confirmation to the user
  await twilioClient.messages.create({
    from: incomingMsg.To, // Your Twilio WhatsApp number
    to: userPhoneNumber,
    body: responseMessage,
  });

  // Acknowledge Twilio's request so it doesn't time out
  res.status(200).send();

  try {
    let analysis;
    let imageBuffer;

    // 1. Check message type and call the correct analysis endpoint
    if (incomingMsg.MediaContentType0 && incomingMsg.MediaUrl0) {
      const mediaUrl = incomingMsg.MediaUrl0;
      const mediaBuffer = await downloadTwilioMedia(mediaUrl);

      const formData = new FormData();

      if (incomingMsg.MediaContentType0.includes("image")) {
        imageBuffer = mediaBuffer; // Keep buffer for logging
        formData.append("image", mediaBuffer, {
          filename: "whatsapp-image.jpg",
          contentType: incomingMsg.MediaContentType0,
        });
        const { data } = await axios.post(`${BASE_URL}/analyze-image`, formData, {
          headers: formData.getHeaders(),
        });
        analysis = data;
      } else if (incomingMsg.MediaContentType0.includes("audio")) {
        formData.append("audio", mediaBuffer, {
          filename: "whatsapp-audio.ogg",
          contentType: incomingMsg.MediaContentType0,
        });
        const { data } = await axios.post(`${BASE_URL}/analyze-audio`, formData, {
          headers: formData.getHeaders(),
        });
        analysis = data;
      }
    } else if (incomingMsg.Body) {
      // It's a text message
      const { data } = await axios.post(`${BASE_URL}/analyze-text`, {
        text: incomingMsg.Body,
      });
      analysis = data;
    }

    if (!analysis) {
        throw new Error("Could not analyze the message.");
    }
    
    // 2. Log the analysis result
    const logFormData = new FormData();
    logFormData.append("userId", userPhoneNumber);
    logFormData.append("userEmail", "xyz@gmail.com"); // Default email as requested
    logFormData.append("analysisResult", JSON.stringify(analysis));

    if (imageBuffer) {
        logFormData.append("image", imageBuffer, {
            filename: "log-image.jpg"
        });
    }

    await axios.post(`${BASE_URL}/log-analysis`, logFormData, {
        headers: logFormData.getHeaders()
    });

    // 3. Send a final confirmation message to the user
    const totalCalories = analysis.details.reduce((sum, item) => sum + (Number(item.calories) || Number(item.calories_burned) || 0), 0);
    responseMessage = `Successfully logged! Total estimated calories: ${Math.round(totalCalories)}.`;

  } catch (error) {
    console.error("Error processing WhatsApp message:", error);
    responseMessage = "Sorry, I couldn't process that. Please try again.";
  }

  // Send final status update
  await twilioClient.messages.create({
    from: incomingMsg.To,
    to: userPhoneNumber,
    body: responseMessage,
  });
});

// --- Existing Root and Listener (Unchanged) ---
app.get("/", (req, res) => {
  res.status(200).json({ status: "healthy", message: "Service is running" });
});

app.listen(port, () => {
  console.log(`✅ Server is running on http://localhost:${port}`);
});