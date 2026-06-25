const http = require('http');
const os = require('os');

const PORT = process.env.PORT || 3000;

// Simple Node Server with an aesthetic dashboard HTML response
const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 3 Node Server Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #6366f1;
            --accent-secondary: #a855f7;
            --success: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            overflow-x: hidden;
        }

        .container {
            max-width: 800px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .pulse {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        .card h3 {
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            font-weight: 400;
        }

        .card .value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .interactive-section {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .interactive-section h2 {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .btn-group {
            display: flex;
            gap: 1rem;
        }

        button {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: opacity 0.2s ease;
        }

        button:hover {
            opacity: 0.9;
        }

        pre {
            background: rgba(0, 0, 0, 0.4);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow-x: auto;
            font-family: monospace;
            font-size: 0.9rem;
            color: #34d399;
            min-height: 80px;
        }

        footer {
            margin-top: 2rem;
            text-align: center;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Day 3 Node Server</h1>
                <p style="color: var(--text-secondary); margin-top: 0.25rem;">Simple & Elegant Node.js Server in Docker</p>
            </div>
            <div class="badge">
                <span class="pulse"></span>
                ONLINE
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <h3>OS Platform</h3>
                <div class="value" id="platform">Loading...</div>
            </div>
            <div class="card">
                <h3>Node.js Version</h3>
                <div class="value" id="node-version">Loading...</div>
            </div>
            <div class="card">
                <h3>Container / Hostname</h3>
                <div class="value" id="hostname">Loading...</div>
            </div>
        </div>

        <div class="interactive-section">
            <h2>Interactive API Console</h2>
            <p style="color: var(--text-secondary);">Query the Node.js backend endpoints directly from here:</p>
            <div class="btn-group">
                <button onclick="fetchAPI('/api/sysinfo')">Get System Info</button>
                <button onclick="fetchAPI('/api/time')">Get Server Time</button>
            </div>
            <pre id="api-output">// Click a button above to run request...</pre>
        </div>

        <footer>
            &copy; 2026 EY Workshop | Created by Jagat
        </footer>
    </div>

    <script>
        function fetchAPI(endpoint) {
            const output = document.getElementById('api-output');
            output.textContent = 'Fetching data...';
            fetch(endpoint)
                .then(res => res.json())
                .then(data => {
                    output.textContent = JSON.stringify(data, null, 2);
                })
                .catch(err => {
                    output.textContent = 'Error: ' + err.message;
                });
        }

        // Initialize values
        fetch('/api/sysinfo')
            .then(res => res.json())
            .then(data => {
                document.getElementById('platform').textContent = data.platform;
                document.getElementById('node-version').textContent = data.nodeVersion;
                document.getElementById('hostname').textContent = data.hostname;
            })
            .catch(() => {
                document.getElementById('platform').textContent = 'N/A';
                document.getElementById('node-version').textContent = 'N/A';
                document.getElementById('hostname').textContent = 'N/A';
            });
    </script>
</body>
</html>
`;

const server = http.createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/html');
    res.end(htmlContent);
  } else if (req.url === '/api/sysinfo') {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({
      status: 'success',
      platform: os.platform(),
      arch: os.arch(),
      release: os.release(),
      hostname: os.hostname(),
      uptime: os.uptime(),
      freeMemoryGB: (os.freemem() / (1024 * 1024 * 1024)).toFixed(2),
      totalMemoryGB: (os.totalmem() / (1024 * 1024 * 1024)).toFixed(2),
      nodeVersion: process.version
    }));
  } else if (req.url === '/api/time') {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({
      status: 'success',
      timestamp: Date.now(),
      isoString: new Date().toISOString(),
      localString: new Date().toString()
    }));
  } else {
    res.statusCode = 404;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Endpoint not found' }));
  }
});

server.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
