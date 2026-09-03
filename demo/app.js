const API_BASE = "/api";

// DOM Elements
const form = document.getElementById("request-form");
const btnScore = document.querySelector(".btn-primary");
const outcomeBanner = document.getElementById("outcome-banner");
const outcomeIcon = document.getElementById("outcome-icon");
const decisionText = document.getElementById("decision-text");
const decisionReason = document.getElementById("decision-reason");
const trustScoreVal = document.getElementById("trust-score-val");
const trustScoreFill = document.getElementById("trust-score-fill");
const semanticVal = document.getElementById("semantic-val");
const emptyVal = document.getElementById("empty-val");
const confVal = document.getElementById("conf-val");
const featureBars = document.getElementById("feature-bars");
const cbAlert = document.getElementById("cb-alert");
const cbText = document.getElementById("cb-text");
const auditIdVal = document.getElementById("audit-id-val");
const returnIdVal = document.getElementById("return-id-val");
const forcedByVal = document.getElementById("forced-by-val");
const customerPhotoPreview = document.getElementById("customer-photo-preview");

// Simulation Toggles
const simTimeout = document.getElementById("sim-timeout");
const simCheckpoint = document.getElementById("sim-checkpoint");

// Setup Simulation Listeners
simTimeout.addEventListener("change", async (e) => {
    if (e.target.checked) {
        simCheckpoint.checked = false;
        await resetFailure();
        triggerFailure("timeout");
    } else {
        resetFailure();
    }
});

simCheckpoint.addEventListener("change", async (e) => {
    if (e.target.checked) {
        simTimeout.checked = false;
        await resetFailure();
        triggerFailure("checkpoint_mismatch");
    } else {
        resetFailure();
    }
});



async function triggerFailure(type) {
    try {
        const fd = new FormData();
        fd.append("failure_type", type);
        await fetch(`${API_BASE}/trigger-failure`, { method: "POST", body: fd });
        showToast(`Simulation Enabled: ${type}`);
    } catch (e) {
        showToast("Error toggling simulation", true);
    }
}

async function resetFailure() {
    try {
        await fetch(`${API_BASE}/reset-failure`, { method: "POST" });
        showToast("Simulations Reset");
    } catch (e) {
        console.error(e);
    }
}

// Generate random realistic data
function randomizeData() {
    // 80% legit, 20% fraud characteristics for fun
    const isFraudulent = Math.random() < 0.2;

    if (isFraudulent) {
        document.getElementById("order_value").value = Math.floor(Math.random() * 20000 + 5000);
        document.getElementById("account_age_days").value = Math.floor(Math.random() * 30 + 1);
        document.getElementById("prior_returns_count").value = Math.floor(Math.random() * 8 + 3);
        document.getElementById("prior_return_approval_rate").value = (Math.random() * 0.4 + 0.6).toFixed(2);
        document.getElementById("return_velocity_7d").value = Math.floor(Math.random() * 4 + 1);
        document.getElementById("is_cod").value = "1";
        document.getElementById("delivery_to_return_hours").value = Math.floor(Math.random() * 24 + 2); // Very fast
    } else {
        document.getElementById("order_value").value = Math.floor(Math.random() * 3000 + 500);
        document.getElementById("account_age_days").value = Math.floor(Math.random() * 700 + 100);
        document.getElementById("prior_returns_count").value = Math.floor(Math.random() * 2);
        document.getElementById("prior_return_approval_rate").value = (Math.random() * 0.2 + 0.8).toFixed(2);
        document.getElementById("return_velocity_7d").value = 0;
        document.getElementById("is_cod").value = Math.random() < 0.5 ? "1" : "0";
        document.getElementById("delivery_to_return_hours").value = Math.floor(Math.random() * 200 + 48); // Normal
    }

    // Randomize item category and distance (Issue #20)
    const categories = ["electronics", "fashion", "home", "beauty", "books", "sports"];
    document.getElementById("item_category").value = categories[Math.floor(Math.random() * categories.length)];
    document.getElementById("address_order_distance_km").value = isFraudulent
        ? (Math.random() * 200 + 50).toFixed(1)
        : (Math.random() * 15 + 1).toFixed(1);

    showToast("Loaded random profile");
}

async function scoreRequest() {
    btnScore.disabled = true;
    btnScore.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" class="spinner"></path></svg> Scoring...`;

    try {
        const formData = new FormData(form);
        // Force the selects to be properly recorded as ints
        formData.set("is_cod", document.getElementById("is_cod").value);
        formData.set("item_category", document.getElementById("item_category").value);


        // Append the uploaded files to formData
        const catalogInput = document.getElementById("catalog_image");
        if (catalogInput && catalogInput.files.length > 0) {
            formData.append("catalog_image", catalogInput.files[0]);
        }

        const returnInput = document.getElementById("return_image");
        if (returnInput && returnInput.files.length > 0) {
            formData.append("return_image", returnInput.files[0]);
        }

        const response = await fetch(`${API_BASE}/score`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        updateUI(data);
    } catch (e) {
        showToast("Error calling API. Is backend running?", true);
        console.error(e);
    } finally {
        btnScore.disabled = false;
        btnScore.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Score Request`;
    }
}

// Turn a raw feature key like "prior_return_approval_rate" into
// "Prior Return Approval Rate" for display.
function formatFeatureName(feat) {
    return feat
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function updateUI(data) {
    // 1. Outcome Banner
    const decision = data.decision.outcome;
    outcomeBanner.className = `outcome-banner glass-panel ${decision}`;

    const icons = {
        'auto_approve': '<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" stroke-width="2" fill="none"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
        'nudge': '<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" stroke-width="2" fill="none"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
        'manual_review': '<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" stroke-width="2" fill="none"><path d="M18 6L6 18M6 6l12 12"></path></svg>'
    };

    outcomeIcon.innerHTML = icons[decision];

    let title = "Approved";
    if (decision === "nudge") title = `Nudge: ${data.decision.nudge_type === 'request_photo' ? 'Request Photo' : 'Offer Store Credit'}`;
    if (decision === "manual_review") title = "Manual Review Required";

    decisionText.innerText = title;
    decisionReason.innerText = data.decision.reason;

    // 2. Scores
    const trust = data.scores.trust_score;
    trustScoreVal.innerText = trust.toFixed(2);

    // Dynamic bar color based on thresholds
    const pct = trust * 100;
    trustScoreFill.style.left = `${pct}%`;
    if (decision === 'auto_approve') trustScoreFill.style.background = 'var(--success)';
    else if (decision === 'nudge') trustScoreFill.style.background = 'var(--warning)';
    else trustScoreFill.style.background = 'var(--danger)';

    // Update threshold markers dynamically on the gauge segments
    const reviewPct = data.thresholds.effective_review * 100;
    const approvePct = data.thresholds.effective_approve * 100;

    const dangerSeg = document.querySelector('.gauge-segment.danger');
    const warningSeg = document.querySelector('.gauge-segment.warning');
    const successSeg = document.querySelector('.gauge-segment.success');

    if (dangerSeg && warningSeg && successSeg) {
        dangerSeg.style.width = `${reviewPct}%`;
        warningSeg.style.width = `${approvePct - reviewPct}%`;
        successSeg.style.width = `${100 - approvePct}%`;
    }

    // Update gauge labels dynamically from API thresholds
    const gaugeLabels = document.querySelectorAll('.gauge-labels span');
    if (gaugeLabels.length === 4) {
        gaugeLabels[0].innerText = '0.0';
        gaugeLabels[1].innerText = data.thresholds.effective_review.toFixed(2);
        gaugeLabels[2].innerText = data.thresholds.effective_approve.toFixed(2);
        gaugeLabels[3].innerText = '1.0';
    }
    if (data.scores.semantic_similarity != null) {
        semanticVal.innerText = data.scores.semantic_similarity.toFixed(2);
    } else {
        semanticVal.innerText = "N/A";
    }

    emptyVal.innerText = data.scores.empty_box_flag == null ? "N/A" : (data.scores.empty_box_flag ? "Yes" : "No");

    if (data.scores.modality_confidence != null) {
        confVal.innerText = data.scores.modality_confidence.toFixed(2);
    } else {
        confVal.innerText = "N/A";
    }

    // 4. Circuit Breaker
    if (data.failure && data.failure.occurred && data.failure.type !== 'image_unavailable') {
        cbAlert.classList.remove("hidden");
        cbText.innerText = `Breaker Tripped: ${data.failure.type}`;
    } else {
        cbAlert.classList.add("hidden");
    }

    // 5. Feature Importance
    let featureHtml = '<div class="feature-bars">';
    const features = data.feature_contributions;
    for (const [feat, val] of Object.entries(features)) {
        const width = Math.max(5, val * 200); // Scale for display
        featureHtml += `
            <div class="feature-row">
                <div class="feature-name" title="${formatFeatureName(feat)}">${formatFeatureName(feat)}</div>
                <div class="feature-bar-container">
                    <div class="feature-bar-fill" style="width: ${width}%"></div>
                </div>
                <div class="feature-val">${(val * 100).toFixed(1)}%</div>
            </div>
        `;
    }
    featureHtml += '</div>';
    featureBars.innerHTML = featureHtml;

    // 6. Audit details
    auditIdVal.innerText = (data.audit_id || 'N/A').split('-')[0] + '...';
    returnIdVal.innerText = data.return_id;
    forcedByVal.innerText = data.decision.forced_by || 'None';
}

function showToast(msg, isError = false) {
    const toast = document.getElementById("toast");
    toast.innerText = msg;
    toast.style.background = isError ? "var(--danger)" : "var(--text-primary)";
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 3000);
}

// Add some basic CSS for spinner dynamically
const style = document.createElement('style');
style.innerHTML = `
@keyframes spin { 100% { transform: rotate(360deg); } }
.spinner { animation: spin 1s linear infinite; transform-origin: center; }
`;
document.head.appendChild(style);

function previewImage(input, previewId) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            const preview = document.getElementById(previewId);
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
        reader.readAsDataURL(input.files[0]);
    }
}