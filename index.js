require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const nodemailer = require("nodemailer");
const supabase = require("./supabaseClient");
const app = express();
app.use(cors());
app.use(express.json());
const port = process.env.PORT || 3000;
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
// Generic multer instance
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
/**
 * @param {string} foodName
 * @returns {Promise<object|null>} The first search result or null.
 */ async function searchFoodSpoonacular(foodName) {
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
 */ async function getNutritionInfoSpoonacular(
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
 */ async function searchRecipeSpoonacular(dishName) {
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
 */ async function getFoodInfoSpoonacular(foodName) {
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
 */ async function getQuickNutritionGuess(foodName) {
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
    // We expect userId and userEmail from the frontend
    const { userId, userEmail, analysisResult } = req.body;
    if (!req.file || !userId || !analysisResult) {
      return res
        .status(400)
        .json({ error: "Image, User ID, and analysis result are required." });
    }
    // 1. Upload the image to Storage
    const fileName = `${Date.now()}-${req.file.originalname}`;
    await supabase.storage
      .from("meal-images")
      .upload(fileName, req.file.buffer, { contentType: req.file.mimetype });
    const { data: urlData } = supabase.storage
      .from("meal-images")
      .getPublicUrl(fileName);
    const imageUrl = urlData.publicUrl;
    // 2. Parse the analysis result
    const parsedAnalysis = JSON.parse(analysisResult);
    const { identifiedFoods, totalEstimatedCalories } = parsedAnalysis;
    // 3. Perform a simple insert into the single 'meals' table
    const { data, error } = await supabase.from("meals").insert([
      {
        user_id: userId,
        user_email: userEmail,
        image_url: imageUrl,
        total_calories: totalEstimatedCalories,
        item_type: "food", // This is always 'food' for this endpoint
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
    // 1. Classify audio as 'food' or 'workout'
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
    // 2. Transcribe the audio
    const transcribeResult = await geminiModel.generateContent([
      "Transcribe this audio.",
      audioPart,
    ]);
    const transcript = transcribeResult.response.text();
    let logDetails = {};
    let totalCalories = 0;
    // 3. Process based on classification
    if (itemType === "workout") {
      // --- NEW: Gemini calorie estimation for workouts ---

      // 1. Create a new prompt asking for a calorie estimate
      const caloriePrompt = `Based on the following workout transcript, provide a rough estimate of the total calories burned. Respond with only a single number. For example: 350. Transcript: "${transcript}"`;
      // 2. Call Gemini with the new prompt
      const calorieResult = await geminiModel.generateContent(caloriePrompt);
      const estimatedCalories =
        parseInt(calorieResult.response.text().trim()) || 0;
      // 3. Update the variables to be saved
      totalCalories = estimatedCalories; // This will now be saved to the database
      logDetails = {
        transcript: transcript,
        estimated_calories_burned: estimatedCalories, // Also save it in the details
      };
    } else {
      // It's food
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
    // 4. Insert into the database
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
// --- UPDATED: The /meals endpoint now does a simple select ---
app.get("/meals", async (req, res) => {
  try {
    const { userId } = req.query;
    if (!userId) {
      return res.status(400).json({ error: "User ID is required." });
    }
    // Simple select query on the single 'meals' table
    const { data, error } = await supabase
      .from("meals")
      .select("*") // Select all columns
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
  const { recipientEmail } = req.body;
  if (!recipientEmail) {
    return res.status(400).json({ error: "Recipient email is required." });
  }
  const mailOptions = {
    to: recipientEmail,
    subject: "You've been invited to view a dashboard!",
    html: ` 
<p>Hello,</p> 
<p>A friend has invited you to view their personal dashboard.</p> 
<p>You can see all their latest activity by visiting this link: <a href="https://your-dashboard-url.com">Your Dashboard Link</a></p> 
<p>Best regards,</p> 
<p>The Dashboard Team</p> 
`,
  };
  try {
    await transporter.sendMail(mailOptions);
    console.log(`Invitation email sent successfully to ${recipientEmail}`);
    res.status(200).json({ message: "Invitation email sent successfully." });
  } catch (error) {
    console.error("Error sending email:", error);
    res.status(500).json({ error: "Failed to send invitation email." });
  }
});
app.get("/", (req, res) => {
  res.status(200).json({ status: "healthy", message: "Service is running" });
});
app.listen(port, () => {
  console.log(`✅ Server is running on http://localhost:${port}`);
});
app.post("/invite-dashboard-access", async (req, res) => {
  const { recipientEmail } = req.body;
  if (!recipientEmail) {
    return res.status(400).json({ error: "Recipient email is required." });
  }
  const mailOptions = {
    to: recipientEmail,
    subject: "You've been invited to view a dashboard!",
    html: `
<p>Hello,</p> 
<p>A friend has invited you to view their personal dashboard.</p> 
<p>You can see all their latest activity by visiting this link: <a href="https://your-dashboard-url.com">Your Dashboard Link</a></p> 
<p>Best regards,</p> 
<p>The Dashboard Team</p> 
`,
  };
  try {
    await transporter.sendMail(mailOptions);
    console.log(`Invitation email sent successfully to ${recipientEmail}`);
    res.status(200).json({ message: "Invitation email sent successfully." });
  } catch (error) {
    console.error("Error sending email:", error);
    res.status(500).json({ error: "Failed to send invitation email." });
  }
});
