/**
 * FastBowl Lab - API Client
 * Handles all communication with the backend server
 */

class FastBowlAPI {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
        this.sessionId = this.getSessionId();
    }

    /**
     * Get or create session ID from URL/storage
     */
    getSessionId() {
        // Try to get from URL params
        const params = new URLSearchParams(window.location.search);
        if (params.has('session')) {
            return params.get('session');
        }
        
        // Try to get from session storage
        if (sessionStorage.getItem('sessionId')) {
            return sessionStorage.getItem('sessionId');
        }
        
        return null;
    }

    /**
     * Create a new session
     */
    async createSession() {
        try {
            const response = await fetch(`${this.baseURL}/session/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            if (data.success) {
                this.sessionId = data.session_id;
                sessionStorage.setItem('sessionId', this.sessionId);
                return data.session_id;
            } else {
                throw new Error(data.error || 'Failed to create session');
            }
        } catch (error) {
            console.error('Create session error:', error);
            throw error;
        }
    }

    /**
     * Get session status
     */
    async getSession() {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(`${this.baseURL}/session/${this.sessionId}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.json();
        } catch (error) {
            console.error('Get session error:', error);
            throw error;
        }
    }

    /**
     * Delete session
     */
    async deleteSession() {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(`${this.baseURL}/session/${this.sessionId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            sessionStorage.removeItem('sessionId');
            this.sessionId = null;
            return await response.json();
        } catch (error) {
            console.error('Delete session error:', error);
            throw error;
        }
    }

    /**
     * Get all previous sessions
     */
    async getSessions() {
        try {
            const response = await fetch(`${this.baseURL}/sessions`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            if (data.success) {
                console.log('✅ Sessions loaded:', data.sessions);
                return data.sessions;
            } else {
                throw new Error(data.error || 'Failed to load sessions');
            }
        } catch (error) {
            console.error('Get sessions error:', error);
            throw error;
        }
    }

    /**
     * Upload video file
     */
    async uploadVideo(file, onProgress = null) {
        if (!this.sessionId) throw new Error('No session ID');
        
        const formData = new FormData();
        formData.append('video', file);
        
        try {
            const xhr = new XMLHttpRequest();
            
            return new Promise((resolve, reject) => {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });
                
                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        const data = JSON.parse(xhr.responseText);
                        if (data.success) {
                            resolve(data);
                        } else {
                            reject(new Error(data.error));
                        }
                    } else {
                        reject(new Error(`HTTP ${xhr.status}`));
                    }
                });
                
                xhr.addEventListener('error', () => {
                    reject(new Error('Upload failed'));
                });
                
                xhr.open('POST', `${this.baseURL}/upload/${this.sessionId}`);
                xhr.send(formData);
            });
        } catch (error) {
            console.error('Upload video error:', error);
            throw error;
        }
    }

    /**
     * Set bowling arm (R or L)
     */
    async setBowlingArm(arm) {
        if (!this.sessionId) throw new Error('No session ID');
        if (!['R', 'L'].includes(arm)) throw new Error('Arm must be R or L');
        
        try {
            const response = await fetch(`${this.baseURL}/bowling-arm/${this.sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ arm })
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.json();
        } catch (error) {
            console.error('Set bowling arm error:', error);
            throw error;
        }
    }

    /**
     * Get calibration preview image
     */
    async getCalibrationPreview() {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(
                `${this.baseURL}/calibration/preview/${this.sessionId}`
            );
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.blob();
        } catch (error) {
            console.error('Get calibration preview error:', error);
            throw error;
        }
    }

    /**
     * Set calibration points
     */
    async setCalibration(topPoint, bottomPoint) {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(
                `${this.baseURL}/calibration/${this.sessionId}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        top: topPoint,
                        bottom: bottomPoint
                    })
                }
            );
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.json();
        } catch (error) {
            console.error('Set calibration error:', error);
            throw error;
        }
    }

    /**
     * Start video processing
     */
    async processVideo() {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(`${this.baseURL}/process/${this.sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.json();
        } catch (error) {
            console.error('Process video error:', error);
            throw error;
        }
    }

    /**
     * Get processing progress
     */
    async getProgress() {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(`${this.baseURL}/progress/${this.sessionId}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.json();
        } catch (error) {
            console.error('Get progress error:', error);
            throw error;
        }
    }

    /**
     * Get all results with detailed error handling
     */
    async getResults() {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(`${this.baseURL}/results/${this.sessionId}`);
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            // Validate response structure
            if (!data.success) {
                throw new Error(data.error || 'No success response');
            }
            
            if (!data.results) {
                throw new Error('No results in response');
            }
            
            // Log result keys for debugging
            console.log('✅ Results received with keys:', Object.keys(data.results));
            
            return data;
        } catch (error) {
            console.error('Get results error:', error);
            throw error;
        }
    }

    /**
     * Get video URL for an analysis type
     */
    getVideoUrl(videoName) {
        if (!this.sessionId) return null;
        return `${this.baseURL}/results/${this.sessionId}/video/${videoName}`;
    }

    /**
     * Get image URL for an analysis type
     */
    getImageUrl(imageName) {
        if (!this.sessionId) return null;
        return `${this.baseURL}/results/${this.sessionId}/image/${imageName}`;
    }

    /**
     * Get specific analysis results
     */
    async getAnalysisResults(analysisType) {
        if (!this.sessionId) throw new Error('No session ID');
        
        const validTypes = ['stride', 'delivery', 'elbow', 'knee', 'com', 'wrist'];
        if (!validTypes.includes(analysisType)) {
            throw new Error(`Invalid analysis type: ${analysisType}`);
        }
        
        try {
            const response = await fetch(
                `${this.baseURL}/results/${this.sessionId}/${analysisType}`
            );
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const result = await response.json();
            if (result.success && result.data) {
                console.log(`✅ ${analysisType} analysis results:`, result.data);
            }
            return result;
        } catch (error) {
            console.error(`Get ${analysisType} results error:`, error);
            throw error;
        }
    }

    /**
     * Get skeletal keypoint reference data
     */
    async getKeypoints() {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            const response = await fetch(`${this.baseURL}/results/${this.sessionId}/keypoints`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const result = await response.json();
            if (result.success && result.data) {
                console.log('✅ Keypoint data received:', result.data);
            }
            return result;
        } catch (error) {
            console.error('Get keypoints error:', error);
            throw error;
        }
    }

    /**
     * Get keypoint frame URL
     */
    getKeypointFrameUrl(frameLabel) {
        if (!this.sessionId) return null;
        return `${this.baseURL}/results/${this.sessionId}/keypoint-frame/${frameLabel}`;
    }

    /**
     * Check API health
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            throw error;
        }
    }

    /**
     * Download analysis files
     */
    async downloadFile(fileType) {
        if (!this.sessionId) throw new Error('No session ID');
        
        try {
            console.log(`⬇️  Downloading: ${fileType}`);
            const url = `${this.baseURL}/results/${this.sessionId}/download/${fileType}`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const blob = await response.blob();
            const contentDisposition = response.headers.get('content-disposition');
            let filename = fileType;
            
            // Extract filename from content-disposition header if available
            if (contentDisposition) {
                const matches = contentDisposition.match(/filename="?([^"]+)"?/);
                if (matches && matches[1]) {
                    filename = matches[1];
                }
            }
            
            // Create download link
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(downloadUrl);
            
            console.log(`✅ Downloaded: ${filename}`);
        } catch (error) {
            console.error(`Download error for ${fileType}:`, error);
            throw error;
        }
    }
}

// Create global API instance and expose to window
const API = new FastBowlAPI();
window.API = API;

// Initialize API logging
console.log('✅ FastBowl API Client initialized');
console.log('Session ID:', window.API.sessionId || 'Not found in URL or storage');

/**
 * Utility functions for detail pages
 */

/**
 * Load detail page data safely
 */
async function loadDetailPageData(metric) {
    try {
        const response = await API.getResults();
        if (!response.success || !response.results) {
            console.error('Failed to load results');
            return null;
        }
        
        const data = response.results[metric];
        if (!data) {
            console.warn(`No data found for metric: ${metric}`);
            return null;
        }
        
        console.log(`✅ Loaded ${metric} data:`, data);
        return data;
    } catch (error) {
        console.error(`Error loading ${metric} data:`, error);
        return null;
    }
}

/**
 * Display loading state
 */
function showLoading(element) {
    if (element) {
        element.innerHTML = '<div class="text-center py-8"><p>Loading analysis data...</p></div>';
    }
}

/**
 * Display error message
 */
function showError(element, message) {
    if (element) {
        element.innerHTML = `<div class="text-center py-8 text-red-400"><p>Error: ${message}</p></div>`;
    }
}

/**
 * Format a number to fixed decimal places
 */
function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return '--';
    }
    return parseFloat(value).toFixed(decimals);
}

/**
 * Get video URL with fallback
 */
function getVideoUrlSafe(videoName) {
    const url = API.getVideoUrl(videoName);
    if (!url) {
        console.warn(`No video URL available for: ${videoName}`);
    }
    return url;
}

/**
 * Get image URL with fallback
 */
function getImageUrlSafe(imageName) {
    const url = API.getImageUrl(imageName);
    if (!url) {
        console.warn(`No image URL available for: ${imageName}`);
    }
    return url;
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FastBowlAPI;
}
