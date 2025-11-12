/**
 * ML2025 Multi-Model App - Frontend JavaScript
 * Handles all AJAX calls and UI updates
 */

// Global state
let dishPredDataLoaded = false;
let dishPredModelTrained = false;
let demandPredDataLoaded = false;
let demandPredModelTrained = false;
let dishRecDataLoaded = false;
let dishRecModelTrained = false;
let prepTimeDataLoaded = false;
let prepTimeModelTrained = false;
let promotionDataLoaded = false;
let promotionModelTrained = false;

// Generated file paths
let generatedDishFile = null;
let generatedDemandFile = null;
let generatedOrderFile = null;
let generatedPrepTimeFile = null;
let generatedPromotionFile = null;

// Utility functions
function showSpinner(spinnerId) {
  document.getElementById(spinnerId).classList.remove("d-none");
}

function hideSpinner(spinnerId) {
  document.getElementById(spinnerId).classList.add("d-none");
}

function showAlert(elementId, message, type = "info") {
  const element = document.getElementById(elementId);
  element.className = `alert alert-${type}`;
  element.textContent = message;
  element.classList.remove("d-none");
}

function hideAlert(elementId) {
  document.getElementById(elementId).classList.add("d-none");
}

// Format number
function formatNumber(num, decimals = 2) {
  return parseFloat(num).toFixed(decimals);
}

function displayCSVPreview(data, previewElementId) {
  const previewDiv = document.getElementById(previewElementId);

  if (!data.sample || data.sample.length === 0) {
    previewDiv.classList.add("d-none");
    return;
  }

  const columns = data.columns || Object.keys(data.sample[0]);

  let html = `
        <div><strong>Preview:</strong> ${data.rows} rows, ${columns.length} columns</div>
        <div class="table-responsive mt-2">
            <table class="table table-sm table-bordered">
                <thead>
                    <tr>
    `;

  columns.forEach((col) => {
    html += `<th>${col}</th>`;
  });

  html += `
                    </tr>
                </thead>
                <tbody>
    `;

  data.sample.forEach((row) => {
    html += "<tr>";
    columns.forEach((col) => {
      const val = row[col] !== null && row[col] !== undefined ? row[col] : "";
      html += `<td>${String(val).substring(0, 50)}</td>`;
    });
    html += "</tr>";
  });

  html += `
                </tbody>
            </table>
        </div>
    `;

  previewDiv.innerHTML = html;
  previewDiv.classList.remove("d-none");
}

// ============================================================================
// DATA GENERATION
// ============================================================================

document
  .getElementById("generateDishData")
  .addEventListener("click", async () => {
    const rows = document.getElementById("dishDataRows").value;
    const btn = document.getElementById("generateDishData");
    const resultDiv = document.getElementById("dishDataGenerated");

    btn.disabled = true;
    resultDiv.classList.add("d-none");

    try {
      const response = await fetch("/api/generate/dish_data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: parseInt(rows) }),
      });

      const result = await response.json();

      if (response.ok) {
        generatedDishFile = result.filepath;
        dishPredDataLoaded = true;
        resultDiv.textContent = `✓ Generated ${result.rows} rows with ${
          result.dishes.length
        } dishes: ${result.dishes.join(", ")}`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "dishPredPreview");
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

document
  .getElementById("generateDemandData")
  .addEventListener("click", async () => {
    const rows = document.getElementById("demandDataRows").value;
    const btn = document.getElementById("generateDemandData");
    const resultDiv = document.getElementById("demandDataGenerated");

    btn.disabled = true;
    resultDiv.classList.add("d-none");

    try {
      const response = await fetch("/api/generate/demand_data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: parseInt(rows) }),
      });

      const result = await response.json();

      if (response.ok) {
        generatedDemandFile = result.filepath;
        demandPredDataLoaded = true;
        resultDiv.textContent = `✓ Generated ${result.rows} rows of demand data`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "demandPredPreview");
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

// document
//   .getElementById("generateOrderData")
//   .addEventListener("click", async () => {
//     const orders = document.getElementById("orderDataRows").value;
//     const btn = document.getElementById("generateOrderData");
//     const resultDiv = document.getElementById("orderDataGenerated");

//     btn.disabled = true;
//     resultDiv.classList.add("d-none");

//     try {
//       const response = await fetch("/api/generate/order_data", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ orders: parseInt(orders) }),
//       });

//       const result = await response.json();

//       if (response.ok) {
//         generatedOrderFile = result.filepath;
//         dishRecDataLoaded = true;
//         resultDiv.textContent = `✓ Generated ${result.orders} orders`;
//         resultDiv.classList.remove("d-none");

//         // Show preview
//         displayCSVPreview(result, "dishRecPreview");
//       } else {
//         alert(`Error: ${result.error}`);
//       }
//     } catch (error) {
//       alert(`Error: ${error.message}`);
//     } finally {
//       btn.disabled = false;
//     }
//   });

// ============================================================================
// USE ORIGINAL DATA
// ============================================================================

document
  .getElementById("useOriginalDishData")
  .addEventListener("click", async () => {
    const btn = document.getElementById("useOriginalDishData");
    const resultDiv = document.getElementById("dishDataGenerated");

    btn.disabled = true;
    resultDiv.classList.add("d-none");

    try {
      const response = await fetch("/api/use_original/dish_prediction");
      const result = await response.json();

      if (response.ok) {
        generatedDishFile = result.filepath;
        dishPredDataLoaded = true;
        resultDiv.textContent = `✓ Loaded original data: ${result.rows} hourly records with ${result.dishes.length} dishes`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "dishPredPreview");
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

document
  .getElementById("useOriginalDemandData")
  .addEventListener("click", async () => {
    const btn = document.getElementById("useOriginalDemandData");
    const resultDiv = document.getElementById("demandDataGenerated");

    btn.disabled = true;
    resultDiv.classList.add("d-none");

    try {
      const response = await fetch("/api/use_original/demand_prediction");
      const result = await response.json();

      if (response.ok) {
        generatedDemandFile = result.filepath;
        demandPredDataLoaded = true;
        resultDiv.textContent = `✓ Loaded original data: ${result.rows} hourly records`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "demandPredPreview");
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

document
  .getElementById("useOriginalOrderData")
  .addEventListener("click", async () => {
    const btn = document.getElementById("useOriginalOrderData");
    const resultDiv = document.getElementById("orderDataGenerated");

    btn.disabled = true;
    resultDiv.classList.add("d-none");

    try {
      const response = await fetch("/api/use_original/dish_recommendation");
      const result = await response.json();

      if (response.ok) {
        generatedOrderFile = result.filepath;
        dishRecDataLoaded = true;
        resultDiv.textContent = `✓ Loaded original data: ${result.orders} orders`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "dishRecPreview");
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

// Update file input handlers to mark data as loaded and show preview
document
  .getElementById("dishPredFile")
  .addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) {
      dishPredDataLoaded = true;

      // Show preview
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("/api/preview_csv", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          displayCSVPreview(result, "dishPredPreview");
        }
      } catch (error) {
        console.error("Preview error:", error);
      }
    }
  });

document
  .getElementById("demandPredFile")
  .addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) {
      demandPredDataLoaded = true;

      // Show preview
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("/api/preview_csv", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          displayCSVPreview(result, "demandPredPreview");
        }
      } catch (error) {
        console.error("Preview error:", error);
      }
    }
  });

document.getElementById("dishRecFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) {
    dishRecDataLoaded = true;

    // Show preview
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/preview_csv", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        displayCSVPreview(result, "dishRecPreview");
      }
    } catch (error) {
      console.error("Preview error:", error);
    }
  }
});

// ============================================================================
// DISH PREDICTION
// ============================================================================

document.getElementById("trainDishPred").addEventListener("click", async () => {
  const fileInput = document.getElementById("dishPredFile");

  // Check if we have data (uploaded file or generated)
  if (!fileInput.files.length && !generatedDishFile) {
    showAlert(
      "dishPredTrainResult",
      "Please upload a CSV file or generate sample data first",
      "danger"
    );
    return;
  }

  const formData = new FormData();

  // Use uploaded file if available, otherwise use generated file
  if (fileInput.files.length) {
    formData.append("file", fileInput.files[0]);
  } else {
    // Fetch generated file and upload it
    const response = await fetch(generatedDishFile);
    const blob = await response.blob();
    formData.append("file", blob, "generated_dish_data.csv");
  }

  showSpinner("trainDishPredSpinner");
  hideAlert("dishPredTrainResult");

  try {
    const response = await fetch("/api/dish_prediction/train", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok) {
      showAlert(
        "dishPredTrainResult",
        "Model trained successfully!",
        "success"
      );
      displayDishPredMetrics(
        result.metrics,
        result.num_dishes,
        result.num_features
      );
      dishPredModelTrained = true;
      document.getElementById("predictDishPred").disabled = false;
    } else {
      showAlert("dishPredTrainResult", `Error: ${result.error}`, "danger");
    }
  } catch (error) {
    showAlert("dishPredTrainResult", `Error: ${error.message}`, "danger");
  } finally {
    hideSpinner("trainDishPredSpinner");
  }
});

function displayDishPredMetrics(metrics, numDishes, numFeatures) {
  const metricsDiv = document.getElementById("dishPredMetrics");

  metricsDiv.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="metric-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3>${formatNumber(metrics.r2_score, 4)}</h3>
                    <p>R² Score</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>${formatNumber(metrics.mae, 2)}</h3>
                    <p>MAE</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3>${formatNumber(metrics.rmse, 2)}</h3>
                    <p>RMSE</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <p><strong>Dishes:</strong> ${numDishes}</p>
            <p><strong>Features:</strong> ${numFeatures}</p>
            <p><strong>Training Samples:</strong> ${metrics.train_samples.toLocaleString()}</p>
            <p><strong>Test Samples:</strong> ${metrics.test_samples.toLocaleString()}</p>
        </div>
    `;
}

document
  .getElementById("predictDishPred")
  .addEventListener("click", async () => {
    const hourInput = document.getElementById("dishPredHour");
    // User enters "hours ahead" (1-24), send as-is to backend
    const hoursAhead = hourInput.value ? parseInt(hourInput.value) : null;

    showSpinner("predictDishPredSpinner");

    try {
      const response = await fetch("/api/dish_prediction/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ hours_ahead: hoursAhead }),
      });

      const result = await response.json();

      if (response.ok) {
        displayDishPredPredictions(result);
      } else {
        document.getElementById(
          "dishPredPredictions"
        ).innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
      }
    } catch (error) {
      document.getElementById(
        "dishPredPredictions"
      ).innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
      hideSpinner("predictDishPredSpinner");
    }
  });

function displayDishPredPredictions(result) {
  const predsDiv = document.getElementById("dishPredPredictions");

  // New format: result contains array of predictions for each hour
  let html = `
        <div class="alert alert-success">
            <strong>Predictions for Next ${result.hours_ahead} Hour(s)</strong><br>
            <small>Total Hours: ${result.total_hours}</small>
        </div>
    `;

  // Display each hour's predictions
  result.predictions.forEach((hourPred, idx) => {
    const totalOrders = hourPred.total_predicted_orders;

    html += `
            <div class="card mb-3">
                <div class="card-header bg-light">
                    <strong class="text-dark">📅 ${hourPred.timestamp}</strong> 
                    <span class="badge bg-primary">${
                      hourPred.hours_ahead
                    } hrs ahead</span>
                    <span class="badge bg-success">${formatNumber(
                      totalOrders,
                      0
                    )} total orders</span>
                </div>
                <div class="card-body p-0">
                    <table class="table table-sm table-hover mb-0">
                        <thead>
                            <tr>
                                <th width="60">Rank</th>
                                <th>Dish</th>
                                <th width="120">Orders</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

    hourPred.dishes.forEach((dish, dishIdx) => {
      html += `
                <tr>
                    <td><strong>${dishIdx + 1}</strong></td>
                    <td>${dish.dish}</td>
                    <td>${formatNumber(dish.predicted_orders, 0)}</td>
                </tr>
            `;
    });

    html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
  });

  predsDiv.innerHTML = html;
}

// ============================================================================
// DEMAND PREDICTION
// ============================================================================

document
  .getElementById("trainDemandPred")
  .addEventListener("click", async () => {
    const fileInput = document.getElementById("demandPredFile");

    // Check if we have data (uploaded file or generated)
    if (!fileInput.files.length && !generatedDemandFile) {
      showAlert(
        "demandPredTrainResult",
        "Please upload a CSV file or generate sample data first",
        "danger"
      );
      return;
    }

    const formData = new FormData();

    // Use uploaded file if available, otherwise use generated file
    if (fileInput.files.length) {
      formData.append("file", fileInput.files[0]);
    } else {
      // Fetch generated file and upload it
      const response = await fetch(generatedDemandFile);
      const blob = await response.blob();
      formData.append("file", blob, "generated_demand_data.csv");
    }

    showSpinner("trainDemandPredSpinner");
    hideAlert("demandPredTrainResult");

    try {
      const response = await fetch("/api/demand_prediction/train", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        showAlert(
          "demandPredTrainResult",
          "Model trained successfully!",
          "success"
        );
        displayDemandPredMetrics(result.metrics, result.num_features);
        demandPredModelTrained = true;
        document.getElementById("predictDemandPred").disabled = false;
      } else {
        showAlert("demandPredTrainResult", `Error: ${result.error}`, "danger");
      }
    } catch (error) {
      showAlert("demandPredTrainResult", `Error: ${error.message}`, "danger");
    } finally {
      hideSpinner("trainDemandPredSpinner");
    }
  });

function displayDemandPredMetrics(metrics, numFeatures) {
  const metricsDiv = document.getElementById("demandPredMetrics");

  metricsDiv.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="metric-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3>${formatNumber(metrics.r2_score, 4)}</h3>
                    <p>R² Score</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>${formatNumber(metrics.mae, 2)}</h3>
                    <p>MAE</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3>${formatNumber(metrics.rmse, 2)}</h3>
                    <p>RMSE</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <p><strong>Features:</strong> ${numFeatures}</p>
            <p><strong>Training Samples:</strong> ${metrics.train_samples.toLocaleString()}</p>
            <p><strong>Test Samples:</strong> ${metrics.test_samples.toLocaleString()}</p>
        </div>
    `;
}

document
  .getElementById("predictDemandPred")
  .addEventListener("click", async () => {
    const hoursInput = document.getElementById("demandPredHours");
    const hours = parseInt(hoursInput.value);

    showSpinner("predictDemandPredSpinner");

    try {
      const response = await fetch("/api/demand_prediction/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ hours }),
      });

      const result = await response.json();

      if (response.ok) {
        displayDemandPredPredictions(result);
      } else {
        document.getElementById(
          "demandPredPredictions"
        ).innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
      }
    } catch (error) {
      document.getElementById(
        "demandPredPredictions"
      ).innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    } finally {
      hideSpinner("predictDemandPredSpinner");
    }
  });

function displayDemandPredPredictions(result) {
  const predsDiv = document.getElementById("demandPredPredictions");

  let html = `
        <div class="alert alert-success">
            <strong>Total Predicted Orders:</strong> ${formatNumber(
              result.total_predicted_orders,
              0
            )}
        </div>
        <div class="predictions-table">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Hour</th>
                        <th>Timestamp</th>
                        <th>Predicted Orders</th>
                    </tr>
                </thead>
                <tbody>
    `;

  result.predictions.forEach((pred) => {
    html += `
            <tr>
                <td><strong>${pred.hour}:00</strong></td>
                <td>${pred.timestamp}</td>
                <td>${formatNumber(pred.predicted_orders, 0)}</td>
            </tr>
        `;
  });

  html += `
                </tbody>
            </table>
        </div>
    `;

  predsDiv.innerHTML = html;
}

// ============================================================================
// DISH RECOMMENDATION
// ============================================================================

document.getElementById("trainDishRec").addEventListener("click", async () => {
  const fileInput = document.getElementById("dishRecFile");

  // Check if we have data (uploaded file or generated)
  if (!fileInput.files.length && !generatedOrderFile) {
    showAlert(
      "dishRecTrainResult",
      "Please upload a CSV file or generate sample data first",
      "danger"
    );
    return;
  }

  const formData = new FormData();

  // Use uploaded file if available, otherwise use generated file
  if (fileInput.files.length) {
    formData.append("file", fileInput.files[0]);
  } else {
    // Fetch generated file and upload it
    const response = await fetch(generatedOrderFile);
    const blob = await response.blob();
    formData.append("file", blob, "generated_order_data.csv");
  }

  showSpinner("trainDishRecSpinner");
  hideAlert("dishRecTrainResult");

  try {
    const response = await fetch("/api/dish_recommend/train", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok) {
      showAlert("dishRecTrainResult", "Model trained successfully!", "success");
      displayDishRecMetrics(result.metrics);
      dishRecModelTrained = true;

      // Load available dishes into dropdown
      await loadAvailableDishes();

      // Enable recommendation inputs
      document.getElementById("dishSelect").disabled = false;
      document.getElementById("topN").disabled = false;
      document.getElementById("getDishRec").disabled = false;
    } else {
      showAlert("dishRecTrainResult", `Error: ${result.error}`, "danger");
    }
  } catch (error) {
    showAlert("dishRecTrainResult", `Error: ${error.message}`, "danger");
  } finally {
    hideSpinner("trainDishRecSpinner");
  }
});

async function loadAvailableDishes() {
  try {
    const response = await fetch("/api/dish_recommend/available_dishes");
    const result = await response.json();

    if (response.ok) {
      const selectElement = document.getElementById("dishSelect");
      selectElement.innerHTML = ""; // Clear existing options

      result.dishes.forEach((dish) => {
        const option = document.createElement("option");
        option.value = dish;
        option.textContent = dish;
        selectElement.appendChild(option);
      });
    }
  } catch (error) {
    console.error("Error loading dishes:", error);
  }
}
function displayDishRecMetrics(metrics) {
  const metricsDiv = document.getElementById("dishRecMetrics");

  metricsDiv.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3>${metrics.num_rules.toLocaleString()}</h3>
                    <p>Association Rules</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>${metrics.num_dishes.toLocaleString()}</h3>
                    <p>Unique Dishes</p>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <p><strong>Transactions:</strong> ${metrics.num_transactions.toLocaleString()}</p>
            <p><strong>Avg Lift:</strong> ${formatNumber(
              metrics.avg_lift,
              2
            )}</p>
            <p><strong>Avg Confidence:</strong> ${formatNumber(
              metrics.avg_confidence * 100,
              2
            )}%</p>
        </div>
    `;
}

// Old search functionality - not needed with dropdown
/*
document.getElementById('searchDish').addEventListener('click', async () => {
    const query = document.getElementById('dishSearch').value;
    
    if (!query) {
        return;
    }
    
    try {
        const response = await fetch('/api/dish_recommend/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayDishSearchResults(result);
        } else {
            document.getElementById('dishSearchResults').innerHTML = 
                `<div class="alert alert-danger">${result.error}</div>`;
        }
    } catch (error) {
        document.getElementById('dishSearchResults').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
});

function displayDishSearchResults(result) {
    const resultsDiv = document.getElementById('dishSearchResults');
    
    if (result.matches.length === 0) {
        resultsDiv.innerHTML = '<div class="alert alert-warning">No dishes found</div>';
        return;
    }
    
    let html = `<p class="text-muted">${result.num_matches} dishes found</p><div class="list-group">`;
    
    result.matches.forEach((dish) => {
        html += `
            <button class="list-group-item list-group-item-action" onclick="document.getElementById('dishName').value='${dish}'; document.getElementById('getDishRec').click();">
                ${dish}
            </button>
        `;
    });
    
    html += '</div>';
    resultsDiv.innerHTML = html;
}
*/

document.getElementById("getDishRec").addEventListener("click", async () => {
  const selectElement = document.getElementById("dishSelect");
  const selectedOptions = Array.from(selectElement.selectedOptions).map(
    (opt) => opt.value
  );
  const topN = parseInt(document.getElementById("topN").value);

  if (selectedOptions.length === 0) {
    document.getElementById("dishRecRecommendations").innerHTML =
      '<div class="alert alert-warning">Please select at least one dish</div>';
    return;
  }

  showSpinner("getDishRecSpinner");

  try {
    // Get recommendations for each selected dish
    const allRecommendations = [];

    for (const dishName of selectedOptions) {
      const response = await fetch("/api/dish_recommend/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ dish_name: dishName, top_n: topN }),
      });

      const result = await response.json();

      if (response.ok) {
        allRecommendations.push({
          dish: dishName,
          recommendations: result.recommendations || [],
        });
      }
    }

    if (allRecommendations.length > 0) {
      displayMultiDishRecommendations(allRecommendations);
    } else {
      document.getElementById("dishRecRecommendations").innerHTML =
        '<div class="alert alert-danger">No recommendations found</div>';
    }
  } catch (error) {
    document.getElementById(
      "dishRecRecommendations"
    ).innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
  } finally {
    hideSpinner("getDishRecSpinner");
  }
});

function displayDishRecRecommendations(result) {
  const recsDiv = document.getElementById("dishRecRecommendations");

  if (result.recommendations.length === 0) {
    recsDiv.innerHTML =
      '<div class="alert alert-warning">No recommendations found</div>';
    return;
  }

  let html = `
        <div class="alert alert-success">
            <strong>Recommendations for:</strong> ${result.query_dish}
        </div>
    `;

  result.recommendations.forEach((rec, idx) => {
    html += `
            <div class="recommendation-item">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <strong>${idx + 1}. ${rec.dish}</strong>
                        <br>
                        <small class="text-muted">Method: ${rec.method}</small>
                    </div>
                    <div class="text-end">
                        <div><strong>${formatNumber(
                          rec.confidence * 100,
                          1
                        )}%</strong></div>
                        <small class="text-muted">Lift: ${formatNumber(
                          rec.lift,
                          2
                        )}x</small>
                    </div>
                </div>
            </div>
        `;
  });

  recsDiv.innerHTML = html;
}

function displayMultiDishRecommendations(allRecommendations) {
  const recsDiv = document.getElementById("dishRecRecommendations");

  let html =
    '<div class="alert alert-success"><strong>Recommendations</strong></div>';

  allRecommendations.forEach((item) => {
    if (item.recommendations.length === 0) {
      html += `
                <div class="card mb-3">
                    <div class="card-header bg-light text-dark">
                        <strong>🍽️ ${item.dish}</strong>
                    </div>
                    <div class="card-body">
                        <p class="text-muted mb-0">No recommendations found</p>
                    </div>
                </div>
            `;
    } else {
      html += `
                <div class="card mb-3">
                    <div class="card-header bg-light text-dark">
                        <strong>🍽️ ${item.dish}</strong>
                    </div>
                    <div class="card-body p-0">
                        <div class="list-group list-group-flush">
            `;

      item.recommendations.forEach((rec, idx) => {
        html += `
                    <div class="list-group-item">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${idx + 1}. ${rec.dish}</strong>
                                <br>
                                <small class="text-muted">${rec.method}</small>
                            </div>
                            <div class="text-end">
                                <div><strong>${formatNumber(
                                  rec.confidence * 100,
                                  1
                                )}%</strong></div>
                                <small class="text-muted">Lift: ${formatNumber(
                                  rec.lift,
                                  2
                                )}x</small>
                            </div>
                        </div>
                    </div>
                `;
      });

      html += `
                        </div>
                    </div>
                </div>
            `;
    }
  });

  recsDiv.innerHTML = html;
}

// ============================================================================
// PREP TIME PREDICTION FUNCTIONS
// ============================================================================

document
  .getElementById("useOriginalPrepTime")
  .addEventListener("click", async () => {
    const btn = document.getElementById("useOriginalPrepTime");
    const resultDiv = document.getElementById("prepTimeTrainingStatus");

    btn.disabled = true;

    try {
      const response = await fetch("/api/use_original/prep_time_prediction");
      const result = await response.json();

      if (response.ok) {
        generatedPrepTimeFile = result.filepath;
        prepTimeDataLoaded = true;
        resultDiv.textContent = `✓ Loaded original data: ${result.rows} orders`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "prepTimePreview");

        // Enable training
        document.getElementById("trainPrepTime").disabled = false;
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

document
  .getElementById("generatePrepTimeData")
  .addEventListener("click", async () => {
    const rows = 1000; // Default rows
    const btn = document.getElementById("generatePrepTimeData");
    const resultDiv = document.getElementById("prepTimeTrainingStatus");

    btn.disabled = true;
    resultDiv.classList.add("d-none");

    try {
      const response = await fetch("/api/generate/prep_time_data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: rows }),
      });

      const result = await response.json();

      if (response.ok) {
        generatedPrepTimeFile = result.filepath;
        prepTimeDataLoaded = true;
        resultDiv.textContent = `✓ Generated ${result.rows} rows of prep time data`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "prepTimePreview");

        // Enable training
        document.getElementById("trainPrepTime").disabled = false;
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

document.getElementById("prepTimeFile").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    prepTimeDataLoaded = true;
    document.getElementById("trainPrepTime").disabled = false;
  }
});

document.getElementById("trainPrepTime").addEventListener("click", async () => {
  const btn = document.getElementById("trainPrepTime");
  const resultDiv = document.getElementById("prepTimeTrainingStatus");

  btn.disabled = true;
  showSpinner("trainPrepTimeSpinner");

  try {
    let formData = new FormData();

    if (generatedPrepTimeFile) {
      // Use generated/original data
      const response = await fetch(`/${generatedPrepTimeFile}`);
      const blob = await response.blob();
      formData.append("file", blob, "data.csv");
    } else {
      // Use uploaded file
      const fileInput = document.getElementById("prepTimeFile");
      formData.append("file", fileInput.files[0]);
    }

    const response = await fetch("/api/prep_time/train", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok) {
      prepTimeModelTrained = true;
      resultDiv.textContent = `✓ Model trained! R²: ${formatNumber(
        result.metrics.r2_score
      )}, MAE: ${formatNumber(result.metrics.mae)} min`;
      resultDiv.classList.remove("d-none");

      // Enable prediction inputs (only kitchen-relevant)
      document.getElementById("prepTimeItems").disabled = false;
      document.getElementById("predictPrepTime").disabled = false;
    } else {
      alert(`Training failed: ${result.error}`);
    }
  } catch (error) {
    alert(`Error: ${error.message}`);
  } finally {
    btn.disabled = false;
    hideSpinner("trainPrepTimeSpinner");
  }
});

document
  .getElementById("predictPrepTime")
  .addEventListener("click", async () => {
    const btn = document.getElementById("predictPrepTime");
    const resultDiv = document.getElementById("prepTimePredictions");

    btn.disabled = true;
    showSpinner("predictPrepTimeSpinner");

    try {
      const orderData = {
        timestamp: new Date().toISOString(),
        "Items in order": document.getElementById("prepTimeItems").value,
        "Order Status": "Delivered",
        // REMOVED: Distance_km, Rider wait time, Subtotal, Total - not kitchen-relevant
      };

      const response = await fetch("/api/prep_time/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(orderData),
      });

      const result = await response.json();

      if (response.ok) {
        resultDiv.innerHTML = `
        <div class="metric-box">
          <h3>${formatNumber(result.predicted_prep_time, 1)}</h3>
          <p>Predicted Prep Time (minutes)</p>
        </div>
        <p><strong>Kitchen-Focused Analysis:</strong></p>
        <ul>
          <li>Items: ${orderData["Items in order"]}</li>
          <li>Order Status: ${orderData["Order Status"]}</li>
          <li><em>Prep time based on order complexity, peak hours, and kitchen factors only</em></li>
        </ul>
      `;
      } else {
        alert(`Prediction failed: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
      hideSpinner("predictPrepTimeSpinner");
    }
  });

// ============================================================================
// PROMOTION EFFECTIVENESS FUNCTIONS
// ============================================================================

document
  .getElementById("useOriginalPromotion")
  .addEventListener("click", async () => {
    const btn = document.getElementById("useOriginalPromotion");
    const resultDiv = document.getElementById("promotionTrainingStatus");

    btn.disabled = true;

    try {
      const response = await fetch("/api/use_original/promotion_effectiveness");
      const result = await response.json();

      if (response.ok) {
        generatedPromotionFile = result.filepath;
        promotionDataLoaded = true;
        resultDiv.textContent = `✓ Loaded original data: ${result.records} records`;
        resultDiv.classList.remove("d-none");

        // Show preview
        displayCSVPreview(result, "promotionPreview");

        // Enable training
        document.getElementById("trainPromotion").disabled = false;
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
    }
  });

// document
//   .getElementById("generatePromotionData")
//   .addEventListener("click", async () => {
//     const rows = 1008; // Default rows
//     const btn = document.getElementById("generatePromotionData");
//     const resultDiv = document.getElementById("promotionTrainingStatus");

//     btn.disabled = true;
//     resultDiv.classList.add("d-none");

//     try {
//       const response = await fetch("/api/generate/promotion_data", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ rows: rows }),
//       });

//       const result = await response.json();

//       if (response.ok) {
//         generatedPromotionFile = result.filepath;
//         promotionDataLoaded = true;
//         resultDiv.textContent = `✓ Generated ${result.rows} rows of promotion data`;
//         resultDiv.classList.remove("d-none");

//         // Show preview
//         displayCSVPreview(result, "promotionPreview");

//         // Enable training
//         document.getElementById("trainPromotion").disabled = false;
//       } else {
//         alert(`Error: ${result.error}`);
//       }
//     } catch (error) {
//       alert(`Error: ${error.message}`);
//     } finally {
//       btn.disabled = false;
//     }
//   });

document.getElementById("promotionFile").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    promotionDataLoaded = true;
    document.getElementById("trainPromotion").disabled = false;
  }
});

document
  .getElementById("trainPromotion")
  .addEventListener("click", async () => {
    const btn = document.getElementById("trainPromotion");
    const resultDiv = document.getElementById("promotionTrainingStatus");

    btn.disabled = true;
    showSpinner("trainPromotionSpinner");

    try {
      let formData = new FormData();

      if (generatedPromotionFile) {
        // Use generated/original data
        const response = await fetch(`/${generatedPromotionFile}`);
        const blob = await response.blob();
        formData.append("file", blob, "data.csv");
      } else {
        // Use uploaded file
        const fileInput = document.getElementById("promotionFile");
        formData.append("file", fileInput.files[0]);
      }

      const response = await fetch("/api/promotion/train", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        promotionModelTrained = true;

        // Display orders metrics if available
        let metricsText = "";
        if (result.metrics.orders && result.metrics.orders.r2 !== undefined) {
          const ordersR2 = formatNumber(result.metrics.orders.r2);
          const ordersMAE = formatNumber(result.metrics.orders.mae);
          const ordersRMSE = formatNumber(result.metrics.orders.rmse);
          metricsText += `Orders R²: ${ordersR2}, MAE: ${ordersMAE}, RMSE: ${ordersRMSE}; `;
        }

        // Display sales metrics
        const salesR2 =
          result.metrics.sales && result.metrics.sales.r2 !== undefined
            ? formatNumber(result.metrics.sales.r2)
            : "N/A";
        const salesMAE =
          result.metrics.sales && result.metrics.sales.mae !== undefined
            ? formatNumber(result.metrics.sales.mae)
            : "N/A";
        const salesRMSE =
          result.metrics.sales && result.metrics.sales.rmse !== undefined
            ? formatNumber(result.metrics.sales.rmse)
            : "N/A";
        metricsText += `Sales R²: ${salesR2}, MAE: ${salesMAE}, RMSE: ${salesRMSE}`;

        resultDiv.textContent = `✓ Model trained! ${metricsText}`;
        resultDiv.classList.remove("d-none");

        // Enable prediction button
        document.getElementById("predictPromotion").disabled = false;
      } else {
        alert(`Training failed: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
      hideSpinner("trainPromotionSpinner");
    }
  });

document
  .getElementById("predictPromotion")
  .addEventListener("click", async () => {
    const btn = document.getElementById("predictPromotion");
    const resultDiv = document.getElementById("promotionPredictions");

    btn.disabled = true;
    showSpinner("predictPromotionSpinner");

    try {
      // Parse the date input
      const startDate = new Date(
        document.getElementById("promotionStartDate").value
      );
      const duration = parseInt(
        document.getElementById("promotionDuration").value
      );

      const promoData = {
        start_date: document.getElementById("promotionStartDate").value,
        duration_days: duration,
        start_hour: parseInt(
          document.getElementById("promotionStartHour").value
        ),
        end_hour: parseInt(document.getElementById("promotionEndHour").value),
        temperature: 25.0, // Default weather
        precipitation: 0.0,
        wind_speed: 5.0,
        is_event: 0,
      };

      // Add promotion flags based on selected type
      const promotionType = document.getElementById("promotionType").value;
      if (promotionType === "discount_10" || promotionType === "discount_20") {
        promoData["flat_%"] = promotionType === "discount_10" ? 10 : 20;
      } else {
        promoData["flat_%"] = 0;
      }

      promoData["flat_rs"] = promotionType === "free_delivery" ? 50 : 0;
      promoData["buy_1_get_1"] = promotionType === "combo_deal" ? 1 : 0;
      promoData["buy_7_get_3"] = 0;

      const response = await fetch("/api/promotion/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(promoData),
      });

      const result = await response.json();

      if (response.ok) {
        let html = `
        <div class="alert alert-success">
          <strong>Promotion Impact Prediction</strong>
        </div>
      `;

        if (
          result.baseline_orders !== undefined &&
          result.baseline_sales !== undefined
        ) {
          html += `
          <div class="metric-box">
            <h3>${formatNumber(result.baseline_orders, 1)}</h3>
            <p>Baseline Orders (${result.total_hours || 1} hours)</p>
          </div>
          <div class="metric-box">
            <h3>₹${formatNumber(result.baseline_sales, 0)}</h3>
            <p>Baseline Sales (${result.total_hours || 1} hours)</p>
          </div>
        `;
        }

        if (result.predicted_orders !== undefined) {
          html += `
          <div class="metric-box">
            <h3>${formatNumber(result.predicted_orders, 1)}</h3>
            <p>Predicted Orders (${
              result.total_hours || 1
            } hours)</p>
          </div>
        `;
        }

        if (result.predicted_sales !== undefined) {
          html += `
          <div class="metric-box">
            <h3>₹${formatNumber(result.predicted_sales, 0)}</h3>
            <p>Predicted Sales (${result.total_hours || 1} hours)</p>
          </div>
        `;
        }

        if (
          result.promotion_impact_orders !== undefined &&
          result.promotion_impact_sales !== undefined
        ) {
          html += `
          <div class="metric-box">
            <h3 style="color: ${
              result.promotion_impact_orders >= 0 ? "green" : "red"
            }">${result.promotion_impact_orders >= 0 ? "+" : ""}${formatNumber(
            result.promotion_impact_orders,
            1
          )}</h3>
            <p>Order Impact</p>
          </div>
          <div class="metric-box">
            <h3 style="color: ${
              result.promotion_impact_sales >= 0 ? "green" : "red"
            }">${result.promotion_impact_sales >= 0 ? "+" : ""}₹${formatNumber(
            result.promotion_impact_sales,
            0
          )}</h3>
            <p>Sales Impact</p>
          </div>
        `;
        }

        html += `
        <p><strong>Promotion Scenario (Period-based, No Historical Data):</strong></p>
        <ul>
      `;

        // Determine promotion type from flags
        let promotionDescription = "No Promotion";
        if (promoData["flat_%"] > 0) {
          promotionDescription = `${promoData["flat_%"]}% Discount`;
        } else if (promoData["flat_rs"] > 0) {
          promotionDescription = `₹${promoData["flat_rs"]} Off`;
        } else if (promoData["buy_1_get_1"] > 0) {
          promotionDescription = "Buy 1 Get 1 Free";
        } else if (promoData["buy_7_get_3"] > 0) {
          promotionDescription = "Buy 7 Get 3 Free";
        }

        html += `<li>Type: ${promotionDescription}</li>`;

        // Show period information if available
        if (promoData.start_date && promoData.duration_days) {
          html += `<li>Period: ${promoData.start_date} for ${promoData.duration_days} days (${promoData.start_hour}:00-${promoData.end_hour}:00 daily)</li>`;
        } else {
          html += `<li>Time: ${promoData.start_hour}:00-${promoData.end_hour}:00</li>`;
        }

        html += `
          <li>Weather: ${promoData.temperature}°C, ${
          promoData.precipitation
        }mm rain</li>
          <li>Events: ${
            promoData.is_event ? "Major event" : "No major events"
          }</li>
          <li><em>Prediction uses volume models trained on aggregated hourly data</em></li>
        </ul>
      `;

        resultDiv.innerHTML = html;
      } else {
        alert(`Prediction failed: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      btn.disabled = false;
      hideSpinner("predictPromotionSpinner");
    }
  });

// Check model status on page load
window.addEventListener("load", async () => {
  try {
    const response = await fetch("/health");
    const result = await response.json();

    console.log("Models status:", result.models);

    // Check promotion model status and enable controls if trained
    if (result.models && result.models.promotion_effectiveness) {
      document.getElementById("predictPromotion").disabled = false;
      console.log("Promotion model already trained - enabling controls");
    }
  } catch (error) {
    console.error("Error checking model status:", error);
  }
});
