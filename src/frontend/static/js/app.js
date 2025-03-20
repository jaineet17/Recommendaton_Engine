/**
 * Amazon Recommendation Engine Frontend
 * JavaScript for interacting with the Recommendation API
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000'; // Update this based on your deployment

// DOM Elements
const statusIndicator = document.querySelector('.status-indicator');
const statusText = document.querySelector('.status-text');
const modelSelect = document.getElementById('model-select');
const userIdInput = document.getElementById('user-id');
const numRecommendationsInput = document.getElementById('num-recommendations');
const includeDetailsCheckbox = document.getElementById('include-details');
const getRecommendationsBtn = document.getElementById('get-recommendations-btn');
const getHistoryBtn = document.getElementById('get-history-btn');
const recommendationsContainer = document.getElementById('recommendations-container');
const historyContainer = document.getElementById('history-container');
const modelInfoText = document.getElementById('model-info-text');
const productModal = document.getElementById('product-modal');
const modalClose = document.querySelector('.close');
const modalProductTitle = document.getElementById('modal-product-title');
const modalProductDetails = document.getElementById('modal-product-details');
const loadingOverlay = document.getElementById('loading-overlay');

// State
let currentUserId = null;
let availableModels = {};

// Event Listeners
document.addEventListener('DOMContentLoaded', init);
getRecommendationsBtn.addEventListener('click', getRecommendations);
getHistoryBtn.addEventListener('click', getUserHistory);
modalClose.addEventListener('click', () => productModal.style.display = 'none');
window.addEventListener('click', (e) => {
    if (e.target === productModal) {
        productModal.style.display = 'none';
    }
});

// Initialization
async function init() {
    try {
        await checkApiStatus();
        await loadAvailableModels();
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Error connecting to the recommendation API. Please try again later.');
    }
}

// API Status Check
async function checkApiStatus() {
    showLoading();
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            const data = await response.json();
            if (data.status === 'healthy') {
                updateStatus(true, 'API Status: Online');
            } else {
                updateStatus(false, 'API Status: Degraded');
            }
        } else {
            updateStatus(false, 'API Status: Offline');
        }
    } catch (error) {
        console.error('API status check error:', error);
        updateStatus(false, 'API Status: Offline');
    } finally {
        hideLoading();
    }
}

// Update status indicator
function updateStatus(isOnline, message) {
    if (isOnline) {
        statusIndicator.classList.remove('offline');
        statusIndicator.classList.add('online');
    } else {
        statusIndicator.classList.remove('online');
        statusIndicator.classList.add('offline');
    }
    statusText.textContent = message;
}

// Load available models
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (!response.ok) {
            throw new Error('Failed to fetch models');
        }
        
        availableModels = await response.json();
        
        // Clear existing options except default
        while (modelSelect.options.length > 1) {
            modelSelect.remove(1);
        }
        
        // Add model options
        for (const [id, name] of Object.entries(availableModels)) {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = name;
            modelSelect.appendChild(option);
        }
        
        // Get system info to display default model
        const sysInfoResponse = await fetch(`${API_BASE_URL}/system-info`);
        if (sysInfoResponse.ok) {
            const sysInfo = await sysInfoResponse.json();
            modelInfoText.textContent = `Default model: ${sysInfo.default_model}`;
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Get recommendations
async function getRecommendations() {
    const userId = userIdInput.value.trim();
    if (!userId) {
        showError('Please enter a user ID.');
        return;
    }
    
    currentUserId = userId;
    const limit = parseInt(numRecommendationsInput.value) || 10;
    const modelName = modelSelect.value;
    const includeDetails = includeDetailsCheckbox.checked;
    
    // Enable history button
    getHistoryBtn.disabled = false;
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommendations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                limit: limit,
                model_name: modelName,
                include_details: includeDetails
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get recommendations');
        }
        
        const data = await response.json();
        displayRecommendations(data);
        modelInfoText.textContent = `Model: ${data.model_info.name} (v${data.model_info.version})`;
    } catch (error) {
        console.error('Error getting recommendations:', error);
        recommendationsContainer.innerHTML = `
            <p class="no-results">Error: ${error.message}</p>
        `;
    } finally {
        hideLoading();
    }
}

// Display recommendations
function displayRecommendations(data) {
    if (!data.recommendations || data.recommendations.length === 0) {
        recommendationsContainer.innerHTML = `
            <p class="no-results">No recommendations found for this user.</p>
        `;
        return;
    }
    
    let html = `<div class="recommendation-list">`;
    
    data.recommendations.forEach(rec => {
        html += `
            <div class="recommendation-item" data-product-id="${rec.product_id}">
                <span class="recommendation-score">${rec.score.toFixed(2)}</span>
                <h3>${rec.title || rec.product_id}</h3>
                ${rec.price ? `<p>Price: $${rec.price.toFixed(2)}</p>` : ''}
                ${rec.category ? `<p>Category: ${rec.category}</p>` : ''}
            </div>
        `;
    });
    
    html += `</div>`;
    recommendationsContainer.innerHTML = html;
    
    // Add click event to recommendation items
    const items = document.querySelectorAll('.recommendation-item');
    items.forEach(item => {
        item.addEventListener('click', () => {
            const productId = item.getAttribute('data-product-id');
            viewProductDetails(productId);
        });
    });
}

// Get user history
async function getUserHistory() {
    if (!currentUserId) {
        showError('No user selected.');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/users/${currentUserId}/history?limit=20`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get user history');
        }
        
        const data = await response.json();
        displayUserHistory(data);
    } catch (error) {
        console.error('Error getting user history:', error);
        historyContainer.innerHTML = `
            <p class="no-results">Error: ${error.message}</p>
        `;
    } finally {
        hideLoading();
    }
}

// Display user history
function displayUserHistory(history) {
    if (!history || history.length === 0) {
        historyContainer.innerHTML = `
            <p class="no-results">No history found for this user.</p>
        `;
        return;
    }
    
    let html = `<div class="history-list">`;
    
    history.forEach(item => {
        let icon = '';
        switch (item.event_type) {
            case 'view':
                icon = '<i class="fas fa-eye history-item-icon"></i>';
                break;
            case 'purchase':
                icon = '<i class="fas fa-shopping-cart history-item-icon"></i>';
                break;
            case 'rate':
                icon = '<i class="fas fa-star history-item-icon"></i>';
                break;
            default:
                icon = '<i class="fas fa-history history-item-icon"></i>';
        }
        
        const timestamp = new Date(item.event_timestamp).toLocaleString();
        
        html += `
            <div class="history-item" data-product-id="${item.product_id}">
                ${icon}
                <div class="history-item-content">
                    <h3>${item.title || item.product_id}</h3>
                    <p>${item.event_type.charAt(0).toUpperCase() + item.event_type.slice(1)}</p>
                </div>
                <span class="history-item-time">${timestamp}</span>
            </div>
        `;
    });
    
    html += `</div>`;
    historyContainer.innerHTML = html;
    
    // Add click event to history items
    const items = document.querySelectorAll('.history-item');
    items.forEach(item => {
        item.addEventListener('click', () => {
            const productId = item.getAttribute('data-product-id');
            viewProductDetails(productId);
        });
    });
}

// View product details
async function viewProductDetails(productId) {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/products/${productId}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get product details');
        }
        
        const product = await response.json();
        
        // Update modal content
        modalProductTitle.textContent = product.title || product.product_id;
        
        let detailsHtml = `
            <p><strong>Product ID:</strong> ${product.product_id}</p>
            <p><strong>Price:</strong> $${product.price ? product.price.toFixed(2) : 'N/A'}</p>
            <p><strong>Category:</strong> ${product.category || 'N/A'}</p>
            <p><strong>Subcategory:</strong> ${product.subcategory || 'N/A'}</p>
            <hr>
            <h3>Description</h3>
            <p>${product.description || 'No description available.'}</p>
        `;
        
        modalProductDetails.innerHTML = detailsHtml;
        
        // Display modal
        productModal.style.display = 'block';
    } catch (error) {
        console.error('Error getting product details:', error);
        showError(`Error loading product details: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Utility Functions
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showError(message) {
    alert(message);
} 