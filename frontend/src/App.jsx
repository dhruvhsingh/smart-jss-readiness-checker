import { useState } from "react";
import "./App.css";

function PoseGuide() {
  return (
    <div className="pose-guide">
      <div className="pose-guide-header">
        <h2>Photo Pose Guide</h2>
        <p>Upload only photos that match this format.</p>
      </div>

      <div className="pose-guide-grid">
        <div className="pose-visual-card">
          <div className="pose-visual">
            <div className="pose-label pose-label-top">Front-facing pose</div>

            <div className="pose-person">
              <div className="pose-head" />
              <div className="pose-body">
                <div className="pose-jacket">Jio jacket worn</div>
              </div>
              <div className="pose-arm pose-arm-left" />
              <div className="pose-arm pose-arm-right" />
              <div className="pose-paper">Laminated Jio paper</div>
              <div className="pose-leg pose-leg-left" />
              <div className="pose-leg pose-leg-right" />
            </div>

            <div className="pose-badge pose-badge-one">One person only</div>
            <div className="pose-badge pose-badge-two">
              Upper body clearly visible
            </div>
          </div>
        </div>

        <div className="pose-rules-card">
          <div className="pose-rule-block good">
            <h3>Do this</h3>
            <ul>
              <li>Exactly one person in frame</li>
              <li>Person should face the camera</li>
              <li>Blue Jio jacket must be worn on body</li>
              <li>Correct laminated Jio promotional paper must be held in hand</li>
              <li>Torso, jacket, and paper should be clearly visible</li>
              <li>Use proper lighting and a clear image</li>
            </ul>
          </div>

          <div className="pose-rule-block bad">
            <h3>Avoid this</h3>
            <ul>
              <li>No group photo or crowd</li>
              <li>No mirror selfie</li>
              <li>No jacket kept on table / hanger / floor</li>
              <li>No paper kept separately or wrong paper</li>
              <li>No blurry, dark, overexposed, or heavily cropped photo</li>
              <li>No side pose where jacket/paper is unclear</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

  const [prmId, setPrmId] = useState("");
  const [photo, setPhoto] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handlePhotoChange = (e) => {
    const file = e.target.files[0];
    setPhoto(file || null);
    setResult(null);
    setError("");

    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
    } else {
      setPreviewUrl("");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!API_BASE_URL) {
      setError("API base URL is not configured.");
      return;
    }

    if (!prmId.trim()) {
      setError("Please enter PRM ID.");
      return;
    }

    if (!photo) {
      setError("Please upload a photo.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("prm_id", prmId);
      formData.append("photo", photo);

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Backend request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1>Smart JSS Readiness Checker</h1>
        <p className="subtitle">
          Upload a Smart JSS photo and verify readiness fields.
        </p>

        <PoseGuide />

        <form onSubmit={handleSubmit} className="form">
          <label htmlFor="prmId">PRM ID</label>
          <input
            id="prmId"
            type="text"
            value={prmId}
            onChange={(e) => setPrmId(e.target.value)}
            placeholder="Enter PRM ID"
          />

          <label htmlFor="photo">Photo</label>
          <input
            id="photo"
            type="file"
            accept="image/*"
            onChange={handlePhotoChange}
          />

          {previewUrl && (
            <div className="preview-box">
              <img src={previewUrl} alt="Preview" className="preview-image" />
            </div>
          )}

          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Photo"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result-box">
            <h2>Result</h2>

            <div className="result-row">
              <span>PRM ID</span>
              <strong>{result.prm_id}</strong>
            </div>

            <div className="result-row">
              <span>Filename</span>
              <strong>{result.filename}</strong>
            </div>

            <div className="result-row">
              <span>Female</span>
              <strong>{result.is_female}</strong>
            </div>

            <div className="result-row">
              <span>Jio Jacket</span>
              <strong>{result.has_jio_jacket}</strong>
            </div>

            <div className="result-row">
              <span>Laminated Jio Promotional Paper</span>
              <strong>{result.has_laminated_jio_promotional_paper}</strong>
            </div>

            <div className="result-row">
              <span>Female Confidence</span>
              <strong>{result.female_confidence}</strong>
            </div>

            <div className="result-row">
              <span>Jacket Confidence</span>
              <strong>{result.jacket_confidence}</strong>
            </div>

            <div className="result-row">
              <span>Paper Confidence</span>
              <strong>{result.paper_confidence}</strong>
            </div>

            <div className="result-row">
              <span>Review Required</span>
              <strong>{result.review_required}</strong>
            </div>

            <div className="result-row review-reason-row">
              <span>Review Reason</span>
              <strong>{result.review_reason || "-"}</strong>
            </div>

            <div className="result-row">
              <span>Timestamp</span>
              <strong>{result.timestamp}</strong>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;