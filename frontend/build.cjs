const fs = require('fs');
const { execSync } = require('child_process');

console.log('[build.cjs] Current working directory:', process.cwd());

if (fs.existsSync('frontend') && fs.existsSync('frontend/package.json')) {
  console.log('[build.cjs] Detected repository root. Preparing frontend...');
  if (!fs.existsSync('frontend/node_modules')) {
    console.log('[build.cjs] Installing frontend dependencies...');
    execSync('npm --prefix frontend install', { stdio: 'inherit' });
  }
  console.log('[build.cjs] Building frontend...');
  execSync('npm --prefix frontend run build', { stdio: 'inherit' });
  
  if (fs.existsSync('frontend/dist')) {
    if (!fs.existsSync('dist')) {
      fs.cpSync('frontend/dist', 'dist', { recursive: true });
    }
  }
} else {
  console.log('[build.cjs] Detected frontend directory. Preparing...');
  if (!fs.existsSync('node_modules')) {
    console.log('[build.cjs] Installing dependencies...');
    execSync('npm install', { stdio: 'inherit' });
  }
  console.log('[build.cjs] Building...');
  execSync('npm run build', { stdio: 'inherit' });
  
  if (fs.existsSync('dist')) {
    if (!fs.existsSync('frontend/dist')) {
      fs.mkdirSync('frontend', { recursive: true });
      fs.cpSync('dist', 'frontend/dist', { recursive: true });
    }
  }
}

console.log('[build.cjs] Build complete.');
