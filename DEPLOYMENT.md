# Streamlit Cloud Deployment Guide

This guide will help you deploy the Solar Challenge Week 1 dashboard to Streamlit Cloud.

## Prerequisites

1. **GitHub Repository**: Your code must be pushed to GitHub
   - Repository: `meleseabrham/solar-challenge-week1`
   - Branch: `main`

2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## Step-by-Step Deployment

### 1. Verify Repository Structure

Ensure your repository has:
```
solar-challenge-week1/
├── app/
│   ├── main.py          ← Main Streamlit app
│   └── utils.py         ← Utility functions
├── requirements.txt      ← Python dependencies
└── data/                ← Data files (optional, can be added later)
```

### 2. Update requirements.txt

Make sure `requirements.txt` includes:
```
numpy
pandas
matplotlib
seaborn
scipy
streamlit
plotly
```

### 3. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)
2. **Click "New app"** or **"Deploy an app"**
3. **Fill in the deployment form**:

   - **Repository**: `meleseabrham/solar-challenge-week1`
   - **Branch**: `main`
   - **Main file path**: `app/main.py` ⚠️ **Use forward slashes, not backslashes!**
   - **App URL** (optional): Leave blank or choose a custom name

4. **Click "Deploy"**

### 4. Important Notes

#### File Path Format
- ✅ **Correct**: `app/main.py` (forward slash)
- ❌ **Wrong**: `app\main.py` (backslash - Windows format)

#### Data Files
The app uses **Streamlit's file uploader** to allow users to upload CSV files directly:
- Users can upload CSV files through the sidebar file uploader
- Supported files: `benin_clean.csv`, `sierraleone_clean.csv`, `togo_clean.csv`
- Files are processed in-memory (no need to store on server)
- The app will automatically detect which country each file belongs to based on the filename

**Note**: Data files are excluded from the repository (see `.gitignore`) to keep the repo size manageable. Users upload their own data files when using the app.

#### Environment Variables (if needed)
If you need environment variables:
1. Go to your app settings in Streamlit Cloud
2. Click "Advanced settings"
3. Add secrets in the "Secrets" section

### 5. Troubleshooting

#### Error: "This file does not exist"
- **Check the path**: Use `app/main.py` (forward slash)
- **Verify the file exists**: Check on GitHub that `app/main.py` is in the repository
- **Check the branch**: Make sure you're deploying from the `main` branch

#### Error: "Module not found"
- **Check requirements.txt**: Ensure all dependencies are listed
- **Wait for installation**: First deployment takes a few minutes to install packages

#### Error: "Data file not found"
- **Add data files to repository**: Commit and push CSV files to the `data/` directory
- **Or use external data source**: Load from URL or cloud storage

### 6. After Deployment

Once deployed:
- Your app will be available at: `https://<your-app-name>.streamlit.app`
- You can share this URL with others
- Updates: Push to GitHub and Streamlit Cloud will auto-deploy

### 7. Updating Your App

1. Make changes to your code locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update dashboard"
   git push origin main
   ```
3. Streamlit Cloud will automatically redeploy (usually takes 1-2 minutes)

## Quick Reference

| Setting | Value |
|---------|-------|
| Repository | `meleseabrham/solar-challenge-week1` |
| Branch | `main` |
| Main file path | `app/main.py` |
| Python version | 3.8+ (auto-detected) |

## Need Help?

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Forum](https://discuss.streamlit.io/)

