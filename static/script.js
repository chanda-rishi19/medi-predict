(() => {
  "use strict";

  const imageInput = document.getElementById("imageInput");
  const preview = document.getElementById("preview");
  const predictBtn = document.getElementById("predictBtn");
  const statusEl = document.getElementById("status");
  const report = document.getElementById("report");
  const visualizations = document.getElementById("visualizations");
  const heatmap = document.getElementById("heatmap");
  const overlay = document.getElementById("overlay");
  let previewUrl = null;

  // Patient-friendly KL descriptions. These explain the grading scale;
  // they are not proof that each listed feature is present in this image.
  const gradeGuidance = {
    0: {
      name: "No radiographic OA",
      meaning: "The X-ray grading criteria do not show definite signs of knee osteoarthritis. This does not rule out pain or other knee problems.",
      why: "The model classified the image as KL Grade 0. Grade 0 generally means there are no definite osteoarthritis features on the X-ray.",
      changes: "No definite osteophytes (bone spurs) or characteristic joint-space changes are expected at this grade.",
      symptoms: "People with Grade 0 may have no symptoms. If you have pain, stiffness, or swelling, it may have another cause or may not be visible on an X-ray.",
      advice: [
        "Keep physically active with comfortable activities such as walking, cycling, or swimming.",
        "Maintain strength in the thigh and hip muscles.",
        "Use sensible training technique and allow recovery after strenuous activity.",
        "Seek an assessment if pain or swelling continues."
      ]
    },
    1: {
      name: "Doubtful / very early OA",
      meaning: "There may be very early changes, but the X-ray findings are uncertain and are not strong enough to confirm definite osteoarthritis.",
      why: "The model classified the image as KL Grade 1. This grade is generally used when there may be a tiny osteophyte or questionable joint-space narrowing.",
      changes: "Possible tiny bone-spur formation or doubtful joint-space narrowing. These subtle findings can be difficult to assess reliably.",
      symptoms: "Some people have no symptoms. Others may notice occasional stiffness or discomfort, but symptoms cannot be determined from the grade alone.",
      advice: [
        "Continue regular low-impact activity as tolerated.",
        "Build thigh and hip strength gradually.",
        "Avoid suddenly increasing exercise intensity if it causes pain.",
        "Discuss persistent symptoms with a healthcare professional."
      ]
    },
    2: {
      name: "Mild radiographic OA",
      meaning: "This grade is generally associated with definite osteophyte formation and possible narrowing of the space between the knee bones.",
      why: "The model classified the image as KL Grade 2. Under the KL scale, definite osteophytes are the key feature; joint-space narrowing may also be present. The model's grade alone does not confirm which feature is visible in your individual X-ray.",
      changes: "Definite osteophytes are characteristic of this grade. There may also be early narrowing of the joint space.",
      symptoms: "Possible symptoms include activity-related knee pain, brief stiffness after rest, or mild swelling. Some people with these X-ray changes have few symptoms.",
      advice: [
        "Do regular strengthening exercises for the quadriceps and hip muscles, ideally with physiotherapy guidance.",
        "Choose low-impact activities such as cycling, swimming, or walking as tolerated.",
        "If you are above a healthy weight, discuss a realistic weight-management plan with a healthcare professional.",
        "Pace activities and avoid repeatedly pushing through significant pain.",
        "Ask a clinician about suitable pain management if symptoms affect daily life."
      ]
    },
    3: {
      name: "Moderate radiographic OA",
      meaning: "The X-ray grading scale describes more established structural changes in the knee joint.",
      why: "The model classified the image as KL Grade 3. This grade is generally associated with multiple osteophytes, definite joint-space narrowing, and some bone hardening (sclerosis).",
      changes: "Typical KL Grade 3 features include multiple bone spurs, definite narrowing of the joint space, some subchondral sclerosis, and possible early change in bone shape.",
      symptoms: "People may experience more frequent pain with walking or stairs, stiffness after rest, swelling, or reduced movement. Symptom severity varies.",
      advice: [
        "Arrange a clinical assessment to discuss symptoms and treatment options.",
        "A physiotherapist can tailor strengthening, flexibility, and mobility exercises.",
        "Use lower-impact activity and adjust movements that significantly aggravate symptoms.",
        "Discuss pain relief and any walking support with a healthcare professional.",
        "If relevant, ask about a sustainable weight-management plan."
      ]
    },
    4: {
      name: "Severe radiographic OA",
      meaning: "This grade represents advanced structural changes associated with knee osteoarthritis on an X-ray.",
      why: "The model classified the image as KL Grade 4. The KL scale generally associates this grade with large osteophytes, marked joint-space narrowing, severe sclerosis, and definite changes in bone shape.",
      changes: "Typical features include large bone spurs, substantial loss of joint space, marked hardening of the bone beneath the cartilage, and altered bone contours.",
      symptoms: "Possible symptoms include persistent pain, stiffness, swelling, reduced movement, and difficulty with walking or everyday activities. X-ray severity and pain do not always match.",
      advice: [
        "Arrange an appointment with an orthopaedic or other qualified healthcare professional.",
        "Discuss a personalised plan that may include physiotherapy, activity modification, and pain management.",
        "Use suitable mobility support if recommended by your clinician.",
        "Joint replacement may be discussed in selected cases based on pain and function—not the X-ray grade alone.",
        "Continue safe movement within your comfort and clinical advice."
      ]
    }
  };

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value || "";
  }

  function getGrade(analysis, payload) {
    const candidates = [analysis?.grade, analysis?.predicted_grade, payload?.grade, payload?.predicted_grade];
    for (const value of candidates) {
      const grade = Number(value);
      if (Number.isInteger(grade) && grade >= 0 && grade <= 4) return grade;
    }
    return null;
  }

  function renderAdvice(items) {
    const list = document.getElementById("advice");
    list.replaceChildren();
    items.forEach((item) => {
      const li = document.createElement("li");
      li.className = "bullet";
      li.textContent = item;
      list.appendChild(li);
    });
  }

  function renderReport(grade, analysis) {
    const info = gradeGuidance[grade];
    setText("gradeTitle", `Grade ${grade}`);
    setText("gradeSubtitle", info.name);
    setText("gradeBadge", `KL Grade ${grade}`);
    setText("meaning", info.meaning);
    setText("why", info.why);
    setText("changes", info.changes);
    setText("symptoms", info.symptoms);
    renderAdvice(info.advice);
    setText(
      "safety",
      "Seek urgent medical assessment if the knee becomes suddenly very painful, hot and swollen, you develop a fever, or you cannot put weight on the leg. For persistent or worsening symptoms, arrange a routine clinical review."
    );

    // Preserve any backend clinical note as additional context, but do not
    // expose model internals, thresholds, preprocessing, or raw console logs.
    const backendNote = analysis?.clinical_note;
    const baseLimitation =
      "This is an AI-generated educational explanation, not a confirmed diagnosis. The predicted grade describes X-ray appearance and does not measure pain or knee function. A qualified healthcare professional should review the image and consider your symptoms and examination.";
    setText("limitation", backendNote ? `${baseLimitation} ${backendNote}` : baseLimitation);
    report.classList.remove("hidden");
    report.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  imageInput.addEventListener("change", () => {
    const file = imageInput.files?.[0];
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (!file) {
      preview.classList.add("hidden");
      preview.removeAttribute("src");
      return;
    }
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.classList.remove("hidden");
    statusEl.textContent = "";
  });

  predictBtn.addEventListener("click", async () => {
    const file = imageInput.files?.[0];
    if (!file) {
      statusEl.textContent = "Please select a knee X-ray image first.";
      return;
    }

    const formData = new FormData();
    // Keep the existing Flask API contract: POST /predict with field "image".
    formData.append("image", file);

    predictBtn.disabled = true;
    predictBtn.textContent = "Analyzing…";
    statusEl.textContent = "Please wait while your X-ray is analyzed.";
    report.classList.add("hidden");
    visualizations.classList.add("hidden");

    try {
      const response = await fetch("/predict", { method: "POST", body: formData });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed. Please try again.");
      }

      const analysis = payload.analysis || {};
      const grade = getGrade(analysis, payload);
      if (grade === null) {
        throw new Error("The server response did not contain a valid grade from 0 to 4. Check the Flask response format.");
      }

      renderReport(grade, analysis);

      // Optional visual explanation. It is deliberately separate from
      // clinical findings and must never be presented as proof of OA features.
      if (payload.heatmap_url || payload.overlay_url) {
        if (payload.heatmap_url) heatmap.src = `${payload.heatmap_url}?t=${Date.now()}`;
        if (payload.overlay_url) overlay.src = `${payload.overlay_url}?t=${Date.now()}`;
        visualizations.classList.remove("hidden");
      }
      statusEl.textContent = "Your patient-friendly report is ready.";
    } catch (error) {
      statusEl.textContent = error.message;
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Analyze X-ray";
    }
  });

  document.getElementById("printBtn").addEventListener("click", () => window.print());
  document.getElementById("newBtn").addEventListener("click", () => {
    imageInput.value = "";
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    report.classList.add("hidden");
    visualizations.classList.add("hidden");
    statusEl.textContent = "Choose another X-ray image to begin.";
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
})(); 
