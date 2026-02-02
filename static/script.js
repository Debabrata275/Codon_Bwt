// Network Background Animation
const canvas = document.getElementById('network-bg');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const particles = [];
const particleCount = window.innerWidth < 768 ? 40 : 80;

class Particle {
  constructor() {
    this.x = Math.random() * canvas.width;
    this.y = Math.random() * canvas.height;
    this.vx = (Math.random() - 0.5) * 0.5;
    this.vy = (Math.random() - 0.5) * 0.5;
    this.radius = Math.random() * 2 + 1;
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;

    if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
    if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
  }

  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(212, 175, 55, 0.5)';
    ctx.fill();
  }
}

for (let i = 0; i < particleCount; i++) {
  particles.push(new Particle());
}

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  particles.forEach(p => {
    p.update();
    p.draw();
  });

  // Draw connections
  particles.forEach((p1, i) => {
    particles.slice(i + 1).forEach(p2 => {
      const dx = p1.x - p2.x;
      const dy = p1.y - p2.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < 150) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = `rgba(212, 175, 55, ${0.2 * (1 - dist / 150)})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
  });

  requestAnimationFrame(animate);
}

animate();

window.addEventListener('resize', () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

// Navigation
function showPage(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-icon').forEach(n => n.classList.remove('active'));
  
  document.getElementById(pageId).classList.add('active');
  event.target.closest('.nav-icon').classList.add('active');
}

// Image Expansion
let expandedImage = null;
const overlay = document.getElementById('overlay');
const imageTitle = document.getElementById('imageTitle');

function expandImage(img) {
  // Collapse previously expanded image if any
  if (expandedImage && expandedImage !== img) {
    expandedImage.classList.remove('expanded');
  }

  // Toggle current image
  if (img.classList.contains('expanded')) {
    img.classList.remove('expanded');
    overlay.classList.remove('active');
    imageTitle.classList.remove('active');
    expandedImage = null;
  } else {
    img.classList.add('expanded');
    overlay.classList.add('active');
    imageTitle.textContent = img.alt;
    imageTitle.classList.add('active');
    expandedImage = img;
  }
}

// Click overlay to close
overlay.addEventListener('click', () => {
  if (expandedImage) {
    expandedImage.classList.remove('expanded');
    overlay.classList.remove('active');
    imageTitle.classList.remove('active');
    expandedImage = null;
  }
});

// Analyze Function with NEW FEATURES
async function analyze() {
  const aa = document.getElementById("aa").value.trim().toUpperCase();
  const codon = document.getElementById("codon").value.trim().toUpperCase();
  const host = document.getElementById("host").value.trim();

  // Show loading
  document.getElementById('loading').style.display = 'block';
  document.getElementById('error-message').style.display = 'none';
  document.getElementById('home-results').style.display = 'none';

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ amino_acid: aa, codon: codon, host_species: host })
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status} ${res.statusText}`);
    }

    const data = await res.json();
    
    // Hide loading
    document.getElementById('loading').style.display = 'none';
    
    if (data.error) {
      document.getElementById('error-message').textContent = data.error;
      document.getElementById('error-message').style.display = 'block';
      return;
    }

    // Show results
    document.getElementById('home-results').style.display = 'block';

    // ========== EXISTING FEATURES ==========
    
    // Codon ranking
    let rHTML = '';
    data.codon_ranking.forEach(r => {
      rHTML += `<tr class="${r.codon === codon ? 'highlight' : ''}">
        <td>${r.rank}</td><td>${r.codon}</td>
        <td>${r.frequency.toFixed(4)}</td>
        <td>${r.ml_weight.toFixed(2)}</td></tr>`;
    });
    document.querySelector("#rankingTable tbody").innerHTML = rHTML;

    // Top species
    document.getElementById("species").innerHTML =
      data.top_species.map((s, index) =>
        `<li class="${index === 0 ? 'top-species' : ''}">${s.SPECIESNAME} — ${s.SCORE.toFixed(4)}</li>`
      ).join("");

    // ========== NEW FEATURE 1: SPECIES-SPECIFIC PREFERENCE ==========
    
    if (data.species_specific_analysis) {
      // High preference species
      document.getElementById("speciesHigh").innerHTML =
        data.species_specific_analysis.top_species.map(s =>
          `<li>${s.SPECIESNAME} <span class="badge">${(s.PREFERENCE_SCORE * 100).toFixed(1)}%</span></li>`
        ).join("");

      // Low preference species
      document.getElementById("speciesLow").innerHTML =
        data.species_specific_analysis.bottom_species.map(s =>
          `<li>${s.SPECIESNAME} <span class="badge">${(s.PREFERENCE_SCORE * 100).toFixed(1)}%</span></li>`
        ).join("");

      // Explanation
      document.getElementById("speciesExplain").innerHTML =
        `<strong>Analysis:</strong> ${data.species_specific_analysis.explanation}`;
    }

    // ========== NEW FEATURE 2: HOST-AWARE OPTIMIZATION ==========
    
    const hostResultDiv = document.getElementById("hostResult");
    const hostRankingContainer = document.getElementById('hostRankingContainer');
    
    if (data.host_aware_optimization) {
      const hostOpt = data.host_aware_optimization;
      hostResultDiv.innerHTML =
        `<strong>Optimal codon for ${hostOpt.host_species}:</strong> 
         <span style="color: #d4af37; font-size: 16px; font-weight: bold;">${hostOpt.optimal_codon}</span>
         <br><br>This codon shows the highest usage frequency in the selected host organism.`;

      // Display host-specific ranking
      if (hostOpt.codon_ranking && hostOpt.codon_ranking.length > 0) {
        hostRankingContainer.style.display = 'block';
        let hostRankHTML = '';
        hostOpt.codon_ranking.forEach(([codon, usage], index) => {
          hostRankHTML += `<tr class="${index === 0 ? 'highlight' : ''}">
            <td>${index + 1}</td>
            <td>${codon}</td>
            <td>${usage.toFixed(4)}</td>
          </tr>`;
        });
        document.querySelector("#hostRankingTable tbody").innerHTML = hostRankHTML;
      } else {
        hostRankingContainer.style.display = 'none';
      }
    } else {
      hostResultDiv.innerHTML =
        `<strong>No host species selected.</strong><br>
         Enter a host species name (e.g., "Escherichia coli") in the input field above to see optimized codon recommendations for that organism.`;
      hostRankingContainer.style.display = 'none';
    }

    // ========== NEW FEATURE 3: CODON BIAS SCORE ==========
    
    const biasResultDiv = document.getElementById("biasResult");
    const biasTableContainer = document.getElementById('biasTableContainer');
    
    if (data.codon_bias_score) {
      const bias = data.codon_bias_score;
      biasResultDiv.innerHTML =
        `<strong>Codon:</strong> ${bias.codon}<br>
         <strong>Global Average Usage:</strong> ${bias.global_average.toFixed(4)}<br><br>
         <strong>Top Biased Species:</strong> These species show the strongest preference for this codon relative to global average.`;

      // Display bias table
      if (bias.top_bias_species && bias.top_bias_species.length > 0) {
        biasTableContainer.style.display = 'block';
        let biasHTML = '';
        bias.top_bias_species.forEach(s => {
          biasHTML += `<tr>
            <td style="text-align: left; padding-left: 12px;">${s.SPECIESNAME}</td>
            <td><span class="badge">${s.bias.toFixed(2)}x</span></td>
          </tr>`;
        });
        document.querySelector("#biasTable tbody").innerHTML = biasHTML;
      } else {
        biasTableContainer.style.display = 'none';
      }
    } else {
      biasResultDiv.innerHTML =
        `<strong>No codon bias data available.</strong><br>
         Enter a specific codon (e.g., "UUA", "UUU") in the codon field above to see bias analysis across species.`;
      biasTableContainer.style.display = 'none';
    }

    // ========== NEW FEATURE 4: CROSS-KINGDOM COMPARISON ==========
    
    const kingdomTableBody = document.querySelector("#kingdomTable tbody");
    
    if (data.cross_kingdom_comparison && data.cross_kingdom_comparison.length > 0) {
      let kingdomHTML = '';
      data.cross_kingdom_comparison.forEach(k => {
        const usageValue = codon && k[codon] !== undefined ? k[codon].toFixed(4) : 'N/A';
        kingdomHTML += `<tr>
          <td style="text-align: left; padding-left: 12px; font-weight: bold;">${k.KINGDOM}</td>
          <td>${usageValue}</td>
        </tr>`;
      });
      kingdomTableBody.innerHTML = kingdomHTML;
    } else {
      if (!codon) {
        kingdomTableBody.innerHTML =
          '<tr><td colspan="2" style="padding: 20px; text-align: center; color: #888;">Enter a codon (e.g., "UUA") in the codon field above to see cross-kingdom comparison.</td></tr>';
      } else {
        kingdomTableBody.innerHTML =
          '<tr><td colspan="2" style="padding: 20px; text-align: center; color: #888;">No KINGDOM data available in the dataset. This feature requires kingdom classification in your CSV file.</td></tr>';
      }
    }

    // ========== EXISTING METRICS ==========

    // Accuracy comparison
    document.getElementById("accCodon").innerText =
      (data.model_metrics.accuracy_codon_only * 100).toFixed(2) + "%";
    document.getElementById("accBWT").innerText =
      (data.model_metrics.accuracy_codon_bwt * 100).toFixed(2) + "%";

    // Metrics table
    const m = data.model_metrics;
    const rows = document.querySelectorAll("#metricsTable tbody tr td:nth-child(2)");
    rows[0].innerText = m.top1_accuracy.toFixed(4);
    rows[1].innerText = m.top2_accuracy.toFixed(4);
    rows[2].innerText = m.top3_accuracy.toFixed(4);
    rows[3].innerText = m.precision.toFixed(4);
    rows[4].innerText = m.recall.toFixed(4);
    rows[5].innerText = m.f1_score.toFixed(4);
    rows[6].innerText = m.loss.toFixed(4);
    rows[7].innerText = (1 - m.top1_accuracy).toFixed(4);

    // Robustness
    document.getElementById("accClean").innerText =
      (m.accuracy_clean * 100).toFixed(2) + "%";
    document.getElementById("accNoisy").innerText =
      (m.accuracy_noisy * 100).toFixed(2) + "%";
    document.getElementById("accMissing").innerText =
      (m.accuracy_missing * 100).toFixed(2) + "%";
    
  } catch (error) {
    // Hide loading
    document.getElementById('loading').style.display = 'none';
    
    // Show error
    document.getElementById('error-message').textContent = 
      `Error: ${error.message}. Make sure Flask server is running on the correct port.`;
    document.getElementById('error-message').style.display = 'block';
    
    console.error('Fetch error:', error);
  }
}

// Test server connection on page load
window.addEventListener('DOMContentLoaded', async () => {
  console.log('Page loaded. Testing server connection...');
  
  try {
    const testRes = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amino_acid: 'L', codon: 'UUA' })
    });
    
    if (testRes.ok) {
      console.log('✓ Server connection successful');
    } else {
      console.warn('⚠ Server returned:', testRes.status, testRes.statusText);
    }
  } catch (err) {
    console.error('✗ Cannot connect to server:', err.message);
    console.error('Make sure Flask is running on http://localhost:5000 or update the fetch URL');
  }
});