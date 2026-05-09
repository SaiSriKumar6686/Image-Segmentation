// SegmentAI — Frontend Logic

document.addEventListener("DOMContentLoaded", () => {
    const uploadZone = document.getElementById("upload-zone");
    const uploadContent = document.getElementById("upload-content");
    const uploadPreview = document.getElementById("upload-preview");
    const previewImg = document.getElementById("preview-img");
    const fileInput = document.getElementById("file-input");
    const uploadBtn = document.getElementById("upload-btn");
    const clearBtn = document.getElementById("clear-btn");
    const actionBar = document.getElementById("action-bar");
    const segmentBtn = document.getElementById("segment-btn");
    const resultsPanel = document.getElementById("results-panel");
    const errorBanner = document.getElementById("error-banner");
    const errorText = document.getElementById("error-text");

    let selectedFile = null;

    // --- File Selection ---
    uploadBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadZone.addEventListener("click", () => {
        if (!selectedFile) fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // --- Drag and Drop ---
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("drag-over");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    // --- Handle File ---
    function handleFile(file) {
        const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp", "image/tiff"];
        if (!validTypes.includes(file.type)) {
            showError("Unsupported file format. Use PNG, JPG, BMP, or WebP.");
            return;
        }
        if (file.size > 16 * 1024 * 1024) {
            showError("File too large. Maximum size is 16 MB.");
            return;
        }

        selectedFile = file;
        hideError();

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadContent.style.display = "none";
            uploadPreview.style.display = "flex";
            actionBar.style.display = "flex";
            resultsPanel.style.display = "none";
        };
        reader.readAsDataURL(file);
    }

    // --- Clear ---
    clearBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        previewImg.src = "";
        uploadContent.style.display = "flex";
        uploadPreview.style.display = "none";
        actionBar.style.display = "none";
        resultsPanel.style.display = "none";
        hideError();
    }

    // --- Segmentation ---
    segmentBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        const btnText = segmentBtn.querySelector(".btn-text");
        const btnLoader = segmentBtn.querySelector(".btn-loader");
        btnText.style.display = "none";
        btnLoader.style.display = "inline-flex";
        segmentBtn.disabled = true;
        hideError();

        try {
            const formData = new FormData();
            formData.append("image", selectedFile);

            const response = await fetch("/api/segment", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || "Segmentation failed");
            }

            displayResults(data);
        } catch (err) {
            showError(err.message || "Failed to connect to the server. Is the backend running?");
        } finally {
            btnText.style.display = "inline";
            btnLoader.style.display = "none";
            segmentBtn.disabled = false;
        }
    });

    // --- Display Results ---
    function displayResults(data) {
        document.getElementById("result-original-img").src = previewImg.src;
        document.getElementById("result-mask-img").src = "data:image/png;base64," + data.mask;
        document.getElementById("result-overlay-img").src = "data:image/png;base64," + data.overlay;

        document.querySelector("#metric-confidence .metric-value").textContent = data.confidence + "%";
        document.querySelector("#metric-coverage .metric-value").textContent = data.coverage + "%";
        document.querySelector("#metric-pixels .metric-value").textContent = data.segmented_pixels.toLocaleString();
        document.querySelector("#metric-total .metric-value").textContent = data.total_pixels.toLocaleString();

        resultsPanel.style.display = "block";
        resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // --- Error handling ---
    function showError(msg) {
        errorText.textContent = msg;
        errorBanner.style.display = "flex";
    }

    function hideError() {
        errorBanner.style.display = "none";
    }

    // --- Smooth scroll for nav links ---
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute("href"));
            if (target) target.scrollIntoView({ behavior: "smooth" });
        });
    });

    // --- Intersection Observer for fade-in ---
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = "1";
                entry.target.style.transform = "translateY(0)";
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll(".feature-card, .tech-item").forEach(el => {
        el.style.opacity = "0";
        el.style.transform = "translateY(24px)";
        el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
        observer.observe(el);
    });

    // --- Navbar scroll effect ---
    window.addEventListener("scroll", () => {
        const navbar = document.getElementById("navbar");
        if (window.scrollY > 20) {
            navbar.style.background = "rgba(14,14,16,0.95)";
        } else {
            navbar.style.background = "rgba(14,14,16,0.8)";
        }
    });
});
