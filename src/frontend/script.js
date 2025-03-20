/**
 * Amazon Recommendation Engine Frontend Script
 * 
 * This script handles all the frontend interactions with the API,
 * including loading models, users, and getting recommendations.
 */

// API Configuration
const API_BASE_URL = 'http://localhost:5050/api';
const API_ENDPOINTS = {
    health: `${API_BASE_URL}/health`,
    models: `${API_BASE_URL}/models`,
    users: `${API_BASE_URL}/users`,
    recommend: (model, userId) => `${API_BASE_URL}/recommend/${model}/${userId}`,
    similarProducts: (model, productId) => `${API_BASE_URL}/similar-products/${model}/${productId}`
};

// DOM Elements
const elements = {
    apiStatus: document.getElementById('api-status'),
    modelSelect: document.getElementById('model-select'),
    modelDetails: document.getElementById('model-details'),
    userSelect: document.getElementById('user-select'),
    productId: document.getElementById('product-id'),
    getRecommendationsBtn: document.getElementById('get-recommendations'),
    getSimilarBtn: document.getElementById('get-similar'),
    recommendationsContainer: document.getElementById('recommendations-container'),
    similarContainer: document.getElementById('similar-container'),
    modelsLoading: document.getElementById('models-loading'),
    usersLoading: document.getElementById('users-loading'),
    recommendationsLoading: document.getElementById('recommendations-loading'),
    similarLoading: document.getElementById('similar-loading')
};

// State
let availableModels = [];
let selectedModel = '';

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventListeners();
});

/**
 * Initialize the application by checking API status and loading initial data
 */
async function initApp() {
    try {
        // Check API health
        const healthResponse = await fetch(API_ENDPOINTS.health);
        const healthData = await healthResponse.json();
        
        if (healthResponse.ok && healthData.status === 'healthy') {
            elements.apiStatus.textContent = 'Connected';
            elements.apiStatus.className = 'badge bg-success';
            
            // Load models and users
            loadModels();
            loadUsers();
        } else {
            elements.apiStatus.textContent = 'API Error';
            elements.apiStatus.className = 'badge bg-danger';
            showError('Failed to connect to the API. Please check if the API server is running.');
        }
    } catch (error) {
        console.error('Error initializing app:', error);
        elements.apiStatus.textContent = 'Connection Failed';
        elements.apiStatus.className = 'badge bg-danger';
        showError('Failed to connect to the API. Please check if the API server is running.');
    }
}

/**
 * Set up event listeners for user interactions
 */
function setupEventListeners() {
    // Model selection change
    elements.modelSelect.addEventListener('change', (e) => {
        selectedModel = e.target.value;
        updateModelDetails(selectedModel);
    });
    
    // Get recommendations button
    elements.getRecommendationsBtn.addEventListener('click', async () => {
        const userId = elements.userSelect.value;
        const model = elements.modelSelect.value;
        
        if (!userId || !model) {
            showError('Please select both a user and a model.');
            return;
        }
        
        await getRecommendations(model, userId);
    });
    
    // Get similar products button
    elements.getSimilarBtn.addEventListener('click', async () => {
        const productId = elements.productId.value.trim();
        const model = elements.modelSelect.value;
        
        if (!productId) {
            showError('Please enter a product ID.');
            return;
        }
        
        if (!model) {
            showError('Please select a model.');
            return;
        }
        
        await getSimilarProducts(model, productId);
    });
}

/**
 * Load available recommendation models from the API
 */
async function loadModels() {
    try {
        elements.modelsLoading.style.display = 'block';
        
        const response = await fetch(API_ENDPOINTS.models);
        const data = await response.json();
        
        if (response.ok) {
            availableModels = data;
            populateModelSelect(data);
        } else {
            showError('Failed to load models.');
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showError('Failed to load models from the API.');
    } finally {
        elements.modelsLoading.style.display = 'none';
    }
}

/**
 * Load available users from the API
 */
async function loadUsers() {
    try {
        elements.usersLoading.style.display = 'block';
        
        const response = await fetch(API_ENDPOINTS.users);
        const data = await response.json();
        
        if (response.ok) {
            populateUserSelect(data.users);
        } else {
            showError('Failed to load users.');
        }
    } catch (error) {
        console.error('Error loading users:', error);
        showError('Failed to load users from the API.');
    } finally {
        elements.usersLoading.style.display = 'none';
    }
}

/**
 * Get recommendations for a user using the specified model
 */
async function getRecommendations(model, userId) {
    try {
        elements.recommendationsLoading.style.display = 'block';
        elements.recommendationsContainer.innerHTML = '';
        
        const response = await fetch(API_ENDPOINTS.recommend(model, userId));
        const data = await response.json();
        
        if (response.ok) {
            displayRecommendations(data);
        } else {
            showError(`Failed to get recommendations: ${data.error}`);
            elements.recommendationsContainer.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        elements.recommendationsContainer.innerHTML = '<p class="text-danger">Error loading recommendations</p>';
    } finally {
        elements.recommendationsLoading.style.display = 'none';
    }
}

/**
 * Get similar products to a given product using the specified model
 */
async function getSimilarProducts(model, productId) {
    try {
        elements.similarLoading.style.display = 'block';
        elements.similarContainer.innerHTML = '';
        
        const response = await fetch(API_ENDPOINTS.similarProducts(model, productId));
        const data = await response.json();
        
        if (response.ok) {
            displaySimilarProducts(data);
        } else {
            showError(`Failed to get similar products: ${data.error}`);
            elements.similarContainer.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Error getting similar products:', error);
        elements.similarContainer.innerHTML = '<p class="text-danger">Error loading similar products</p>';
    } finally {
        elements.similarLoading.style.display = 'none';
    }
}

/**
 * Populate the model selection dropdown
 */
function populateModelSelect(models) {
    elements.modelSelect.innerHTML = '';
    
    // Default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a model';
    elements.modelSelect.appendChild(defaultOption);
    
    // Add each model as an option
    Object.keys(models).forEach(modelName => {
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName.toUpperCase();
        elements.modelSelect.appendChild(option);
    });
    
    // Set the first model as selected if available
    if (Object.keys(models).length > 0) {
        elements.modelSelect.value = Object.keys(models)[0];
        selectedModel = Object.keys(models)[0];
        updateModelDetails(selectedModel);
    }
}

/**
 * Populate the user selection dropdown
 */
function populateUserSelect(users) {
    elements.userSelect.innerHTML = '';
    
    // Default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a user';
    elements.userSelect.appendChild(defaultOption);
    
    // Add each user as an option
    users.forEach(userId => {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = userId;
        elements.userSelect.appendChild(option);
    });
    
    // Set the first user as selected if available
    if (users.length > 0) {
        elements.userSelect.value = users[0];
    }
}

/**
 * Update the model details section
 */
function updateModelDetails(modelName) {
    if (!modelName || !availableModels[modelName]) {
        elements.modelDetails.innerHTML = '<p>Select a model to see details</p>';
        return;
    }
    
    const model = availableModels[modelName];
    
    const html = `
        <h6>${modelName.toUpperCase()}</h6>
        <p><strong>Version:</strong> ${model.version}</p>
        <p><strong>Training Date:</strong> ${model.training_date}</p>
        <p><strong>Users:</strong> ${model.num_users} | <strong>Products:</strong> ${model.num_products}</p>
        <div class="mt-2">
            <h6>Performance Metrics:</h6>
            <ul class="list-unstyled">
                ${Object.entries(model.metrics).map(([key, value]) => 
                    `<li><small>${key}: <span class="fw-bold">${typeof value === 'number' ? value.toFixed(4) : value}</span></small></li>`
                ).join('')}
            </ul>
        </div>
    `;
    
    elements.modelDetails.innerHTML = html;
}

/**
 * Display recommendations in the recommendations container
 */
function displayRecommendations(data) {
    const recommendations = data.recommendations;
    
    if (!recommendations || recommendations.length === 0) {
        elements.recommendationsContainer.innerHTML = '<p>No recommendations found for this user.</p>';
        return;
    }
    
    let html = `
        <h5 class="mb-3">Recommendations for ${data.user_id}</h5>
        <div class="recommendations-list">
    `;
    
    recommendations.forEach((productId, index) => {
        html += `
            <div class="product-card">
                <span class="product-id">${index + 1}. ${productId}</span>
            </div>
        `;
    });
    
    html += '</div>';
    elements.recommendationsContainer.innerHTML = html;
}

/**
 * Display similar products in the similar products container
 */
function displaySimilarProducts(data) {
    const similarProducts = data.similar_products;
    
    if (!similarProducts || similarProducts.length === 0) {
        elements.similarContainer.innerHTML = '<p>No similar products found.</p>';
        return;
    }
    
    let html = `
        <h5 class="mb-3">Products similar to ${data.product_id}</h5>
        <div class="similar-products-list">
    `;
    
    similarProducts.forEach((productId, index) => {
        html += `
            <div class="product-card">
                <span class="product-id">${index + 1}. ${productId}</span>
            </div>
        `;
    });
    
    html += '</div>';
    elements.similarContainer.innerHTML = html;
}

/**
 * Show an error message
 */
function showError(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.querySelector('.container').prepend(alertDiv);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        const bsAlert = new bootstrap.Alert(alertDiv);
        bsAlert.close();
    }, 5000);
} 