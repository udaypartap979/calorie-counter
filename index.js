require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const { GoogleGenerativeAI } = require("@google/generative-ai"); // Import Gemini

// test comment 
// test commit ishwar_test_branch
const app = express();
app.use(cors());
const port = process.env.PORT || 3000;

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-1.5-pro-latest" });

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
});


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
          apiKey: process.env.SPOONACULAR_API_KEY
        }
      }
    );

    if (response.data.results && response.data.results.length > 0) {
      return response.data.results[0];
    }
    return null;
  } catch (error) {
    console.error(`Error searching food in Spoonacular: ${foodName}:`, error.message);
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
async function getNutritionInfoSpoonacular(ingredientId, amount = 100, unit = "grams") {
  try {
    const response = await axios.get(
      `https://api.spoonacular.com/food/ingredients/${ingredientId}/information`,
      {
        params: {
          amount: amount,
          unit: unit,
          apiKey: process.env.SPOONACULAR_API_KEY
        }
      }
    );

    if (response.data && response.data.nutrition) {
      const nutrition = response.data.nutrition;
      const calories = nutrition.nutrients.find(n => n.name === "Calories");
      
      return {
        name: response.data.name,
        calories: calories ? Math.round(calories.amount) : 0,
        serving_size: `${amount} ${unit}`,
        weight_grams: amount,
        nutrition: {
          protein: nutrition.nutrients.find(n => n.name === "Protein")?.amount || 0,
          fat: nutrition.nutrients.find(n => n.name === "Fat")?.amount || 0,
          carbs: nutrition.nutrients.find(n => n.name === "Carbohydrates")?.amount || 0,
          fiber: nutrition.nutrients.find(n => n.name === "Fiber")?.amount || 0,
          sugar: nutrition.nutrients.find(n => n.name === "Sugar")?.amount || 0
        }
      };
    }
    return null;
  } catch (error) {
    console.error(`Error getting nutrition info from Spoonacular:`, error.message);
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
          apiKey: process.env.SPOONACULAR_API_KEY
        }
      }
    );

    if (response.data.results && response.data.results.length > 0) {
      const recipe = response.data.results[0];
      const nutrition = recipe.nutrition;
      
      if (nutrition && nutrition.nutrients) {
        const calories = nutrition.nutrients.find(n => n.name === "Calories");
        
        return {
          name: recipe.title,
          calories: calories ? Math.round(calories.amount) : 0,
          serving_size: `1 serving (${recipe.servings} total servings)`,
          servings: recipe.servings,
          nutrition: {
            protein: nutrition.nutrients.find(n => n.name === "Protein")?.amount || 0,
            fat: nutrition.nutrients.find(n => n.name === "Fat")?.amount || 0,
            carbs: nutrition.nutrients.find(n => n.name === "Carbohydrates")?.amount || 0,
            fiber: nutrition.nutrients.find(n => n.name === "Fiber")?.amount || 0,
            sugar: nutrition.nutrients.find(n => n.name === "Sugar")?.amount || 0
          },
          source: "recipe"
        };
      }
    }
    return null;
  } catch (error) {
    console.error(`Error searching recipe in Spoonacular: ${dishName}:`, error.message);
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
      const nutritionInfo = await getNutritionInfoSpoonacular(ingredient.id, 100, "grams");
      if (nutritionInfo) {
        return {
          ...nutritionInfo,
          source: "ingredient"
        };
      }
    }

    const recipeInfo = await searchRecipeSpoonacular(foodName);
    if (recipeInfo) {
      return recipeInfo;
    }

    return null;
  } catch (error) {
    console.error(`Error getting food info from Spoonacular: ${foodName}:`, error.message);
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
          apiKey: process.env.SPOONACULAR_API_KEY
        }
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
          carbs: Math.round(response.data.carbs?.value || 0)
        },
        source: "nutrition_guess"
      };
    }
    return null;
  } catch (error) {
    console.error(`Error getting nutrition guess from Spoonacular: ${foodName}:`, error.message);
    return null;
  }
}


app.post("/identify-food", upload.single("foodImage"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file provided." });
    }

    console.log("Step 1: Identifying food items with Gemini 1.5 Pro...");

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

    const identifiedFoods = geminiResponseText.split(',').map(item => item.trim()).filter(item => item);

    console.log("Analysis complete (Gemini 1.5 Pro):", identifiedFoods);
    
    if (identifiedFoods.length === 0) {
        return res.status(404).json({ error: "No food items could be identified." });
    }
    
    console.log(`Step 2: Fetching nutrition for: ${identifiedFoods.join(', ')}`);
    const foodsWithNutrition = await Promise.all(
      identifiedFoods.map(async (food) => {
        let nutritionInfo = await getFoodInfoSpoonacular(food) || await getQuickNutritionGuess(food);
        
        if (nutritionInfo) {
          return {
            name: nutritionInfo.name,
            calories: nutritionInfo.calories,
            serving_size: nutritionInfo.serving_size,
            nutrition: nutritionInfo.nutrition,
            source: nutritionInfo.source
          };
        } else {
          return { name: food, calories: "Unknown", nutrition: null, source: "not_found" };
        }
      })
    );

    const totalCalories = foodsWithNutrition.reduce((sum, food) => sum + (Number(food.calories) || 0), 0);
    
    res.status(200).json({
      identifiedFoods: foodsWithNutrition,
      totalEstimatedCalories: totalCalories,
      note: "Food identification by Gemini 1.5 Pro. Nutrition values are estimates provided by Spoonacular API."
    });

  } catch (error) {
    console.error("ERROR during image analysis:", error);
    res.status(500).json({ error: "Failed to analyze image." });
  }
});

app.get("/", (req, res) => {
  res.status(200).json({ status: "healthy", message: "Service is running" });
});

app.listen(port, () => {
  console.log(`✅ Server is running on http://localhost:${port}`);
});