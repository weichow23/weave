// WEAVE Data Viewer
// A simple viewer to explore WEAVE dataset samples by domain

class WeaveDataViewer {
    constructor() {
        this.currentDomain = null;
        this.domainData = {
            "edit": { count: 13, samples: [] },
            "recall": { count: 12, samples: [] },
            "fusion": { count: 12, samples: [] },
            "story": { count: 10, samples: [] },
            "chemistry": { count: 7, samples: [] },
            "geography": { count: 6, samples: [] },
            "optics": { count: 6, samples: [] },
            "visual_jigsaw": { count: 5, samples: [] },
            "spatial": { count: 5, samples: [] },
            "physics": { count: 5, samples: [] },
            "minecraft": { count: 5, samples: [] },
            "chess_game": { count: 4, samples: [] },
            "mathematics": { count: 3, samples: [] },
            "astronomy": { count: 3, samples: [] },
            "maze": { count: 2, samples: [] },
            "biology": { count: 2, samples: [] }
        };
        
        // Sample data for each domain (simplified placeholder examples)
        // In a real implementation, these would be loaded from a server
        this.loadSampleData();
        
        // Initialize the viewer
        this.initViewer();
    }
    
    async loadSampleData() {
        try {
            const response = await fetch('static/test.json');
            if (response.ok) {
                const allData = await response.json();
                
                // Group samples by domain
                for (const sample of allData) {
                    const domain = sample.domain;
                    if (this.domainData[domain]) {
                        this.domainData[domain].samples.push(sample);
                    }
                }
                
                console.log('Successfully loaded all sample data');
            } else {
                console.error(`Failed to load data: ${response.status}`);
            }
        } catch (error) {
            console.error("Error loading sample data:", error);
        }
    }
    
    generateSampleData(domain, index) {
        // Generate a sample data object for visualization
        // In a real implementation, these would be actual dataset samples
        const images = [];
        const randomImageCount = 4 + Math.floor(Math.random() * 4); // 4-7 images
        
        for (let i = 0; i < randomImageCount; i++) {
            // Use placeholder images based on domain
            images.push(`static/images/sample/${domain}_${i % 3 + 1}.jpg`);
        }
        
        const chats = [];
        const chatTurns = 3 + Math.floor(Math.random() * 3); // 3-5 chat turns
        
        for (let i = 0; i < chatTurns; i++) {
            if (i % 2 === 0) {
                // User turn
                chats.push({
                    role: "user",
                    type: "text",
                    content: this.getUserPrompt(domain, i),
                    key_point: this.getKeyPoint(domain, i)
                });
            } else {
                // If this is a response that includes an image
                if (i < chatTurns - 1 || Math.random() > 0.5) {
                    chats.push({
                        role: "assistant",
                        type: "image",
                        content: `Generated image #${Math.floor(i/2) + 1}`,
                        key_point: null
                    });
                }
                
                chats.push({
                    role: "assistant",
                    type: "text",
                    content: this.getAssistantResponse(domain, i),
                    key_point: null
                });
            }
        }
        
        return {
            domain: domain,
            images: images,
            chats: chats
        };
    }
    
    getUserPrompt(domain, turn) {
        // Generate domain-specific placeholder prompts
        const domainPrompts = {
            "edit": [
                "Can you modify this image by making the sky more dramatic with storm clouds?",
                "Now change the colors to be more vibrant and saturated",
                "Add a small boat on the water in the distance"
            ],
            "recall": [
                "Do you remember what was in the first image I showed you?",
                "Can you recreate that first image but add a mountain in the background?",
                "Now combine elements from the first and second images"
            ],
            "fusion": [
                "Can you combine these two images into one coherent scene?",
                "Add elements from the previous result to this new image",
                "Now give everything a winter theme with snow"
            ],
            "chemistry": [
                "Can you show me the molecular structure of benzene?",
                "Now show me a reaction between benzene and chlorine",
                "Create a diagram showing orbital hybridization in this molecule"
            ],
            "geography": [
                "Can you create a map showing the major mountain ranges in Asia?",
                "Now highlight the regions with Mediterranean climate",
                "Add major river systems to the map"
            ],
            "physics": [
                "Can you illustrate the principle of conservation of momentum?",
                "Now show what happens in an elastic collision between two objects",
                "Create a diagram of a simple electric circuit"
            ],
            "visual_jigsaw": [
                "Can you arrange these visual elements to complete the pattern?",
                "Now rotate the central element 90 degrees clockwise",
                "Replace the top-right element with something that fits the pattern"
            ],
            // Default prompts for other domains
            "default": [
                "Can you generate an image related to " + domain + "?",
                "Now modify the previous result by adding more detail",
                "Create a variation with different lighting"
            ]
        };
        
        const promptSet = domainPrompts[domain] || domainPrompts.default;
        return promptSet[turn % promptSet.length] + (turn > 0 ? " Remember what we discussed earlier." : "");
    }
    
    getAssistantResponse(domain, turn) {
        // Generate domain-specific placeholder responses
        return `Here's the result based on your request related to ${domain}. I've applied the changes you requested${turn > 0 ? " while maintaining consistency with our previous work" : ""}.`;
    }
    
    getKeyPoint(domain, turn) {
        // Generate domain-specific placeholder key points
        if (domain === "edit") {
            return {
                "modification_type": "visual enhancement",
                "elements_changed": "lighting, color, composition",
                "context_preserved": "Yes, original scene is still recognizable"
            };
        } else if (domain === "recall") {
            return {
                "memory_accuracy": "high",
                "elements_recalled": "main subject, background, color scheme",
                "modifications_applied": "as specified in prompt"
            };
        } else {
            return {
                "task_type": domain,
                "complexity": "medium",
                "key_challenge": "maintaining visual coherence across changes"
            };
        }
    }
    
    initViewer() {
        // Create domain buttons
        const domainButtonsContainer = document.getElementById('weave-domain-buttons');
        if (!domainButtonsContainer) {
            console.error('Domain buttons container not found');
            return;
        }
        
        Object.entries(this.domainData).forEach(([domain, data]) => {
            const button = document.createElement('button');
            button.className = 'domain-button';
            button.setAttribute('data-domain', domain);
            
            // Remove the count display, show only the domain name
            button.innerHTML = `${domain.replace('_', ' ')}`;
            
            button.addEventListener('click', () => this.showDomainSample(domain));
            domainButtonsContainer.appendChild(button);
        });
        
        // Create viewer container
        const viewerContainer = document.getElementById('weave-data-sample');
        if (!viewerContainer) {
            console.error('Viewer container not found');
            return;
        }
    }
    
    showDomainSample(domain) {
        this.currentDomain = domain;
        
        // Highlight the selected domain button
        document.querySelectorAll('.domain-button').forEach(button => {
            if (button.getAttribute('data-domain') === domain) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
        
        // Get a random sample from the domain
        const samples = this.domainData[domain].samples;
        const randomSample = samples[Math.floor(Math.random() * samples.length)];
        
        // Display the sample
        this.renderSample(randomSample);
    }
    
    renderSample(sample) {
        const container = document.getElementById('weave-data-sample');
        if (!container) return;
        
        // Clear previous content
        container.innerHTML = '';
        
        // Create domain header
        const domainHeader = document.createElement('div');
        domainHeader.className = 'domain-header';
        domainHeader.innerHTML = `<h4 class="domain-title">Domain: ${sample.domain.replace('_', ' ').toUpperCase()}</h4>`;
        container.appendChild(domainHeader);
        
        // Create images section
        const imagesSection = document.createElement('div');
        imagesSection.className = 'images-section';
        imagesSection.innerHTML = `
            <h4 class="section-title">🖼️ Images (${sample.images.length})</h4>
            <div class="images-grid"></div>
        `;
        container.appendChild(imagesSection);
        
        const imagesGrid = imagesSection.querySelector('.images-grid');
        sample.images.forEach((imagePath, index) => {
            const imageItem = document.createElement('div');
            imageItem.className = 'image-item';
            const adjustedPath = imagePath.replace('imgs/', 'static/images/sample/');
            imageItem.innerHTML = `
                <div class="image-container">
                    <img src="${adjustedPath}" alt="Image #${index + 1}" class="sample-image" onerror="this.src='static/images/placeholder.jpg'">
                </div>
                <div class="image-label">Image #${index + 1}</div>
            `;
            imagesGrid.appendChild(imageItem);
        });
        
        // Create conversation section
        const chatSection = document.createElement('div');
        chatSection.className = 'chat-section';
        chatSection.innerHTML = `
            <h4 class="section-title">💬 Conversation (${sample.chats.length} turns)</h4>
            <div class="chat-container"></div>
        `;
        container.appendChild(chatSection);
        
        const chatContainer = chatSection.querySelector('.chat-container');
        sample.chats.forEach((chat, index) => {
            const chatItem = document.createElement('div');
            chatItem.className = `chat-item role-${chat.role}`;
            
            let contentHtml;
            if (chat.type === 'image') {
                contentHtml = `<div class="chat-image-reference">${chat.content}</div>`;
            } else {
                contentHtml = `<div class="chat-text">${chat.content}</div>`;
            }
            
            chatItem.innerHTML = `
                <div class="chat-header">
                    <span class="chat-role">${chat.role.toUpperCase()}</span>
                    <span class="chat-type">${chat.type.toUpperCase()}</span>
                </div>
                <div class="chat-content">
                    ${contentHtml}
                </div>
            `;
            
            chatContainer.appendChild(chatItem);
        });
        
        // Show the random button at the bottom
        const randomButton = document.createElement('div');
        randomButton.className = 'random-sample-button';
        randomButton.innerHTML = `
            <button id="next-sample">Show Another ${sample.domain.replace('_', ' ')} Sample</button>
        `;
        container.appendChild(randomButton);
        
        // Add event listener to the random button
        document.getElementById('next-sample').addEventListener('click', () => {
            this.showDomainSample(sample.domain);
        });
    }
}

// Initialize the viewer when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if the viewer containers exist
    if (document.getElementById('weave-domain-buttons') && document.getElementById('weave-data-sample')) {
        window.weaveDataViewer = new WeaveDataViewer();
    }
});
