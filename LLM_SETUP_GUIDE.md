# LLM Setup Guide

Enable AI-powered clustering assistance in Clustery using OpenAI. No programming experience needed!

## Easy Setup (Recommended)

### Step 1: Get Your API Key

Go to OpenAI platform (platform.openai.com/api-keys) and create a new secret key.

### Step 2: Run Setup Script

Open your terminal and run:

```bash
python setup_api_keys.py
```

Enter your API key when prompted. The script will create a configuration file for you automatically.

The setup script will automatically test your API key to make sure it's working.

### Step 3: Launch Clustery

```bash
streamlit run main.py
```

Your AI features are now ready to use!

## Using AI Assist

1. Complete clustering in the app
2. Navigate to **Fine-tuning** tab
3. Scroll to **AI Assist** section
4. Select provider (OpenAI) and model
5. Choose operation:
   - **Suggest Cluster Names**: Better cluster naming
   - **Suggest Entry Moves**: Identify misplaced entries
   - **Suggest Merges/Splits**: Optimize cluster structure
   - **Free-form Query**: General clustering help

## Alternative: Environment Variables

If you prefer using environment variables:

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="sk-your_key_here"
```

**Linux/Mac:**

```bash
export OPENAI_API_KEY="sk-your_key_here"
```

## Supported Models

- `gpt-4o` - Latest GPT-4
- `gpt-4o-mini` - Cheaper, faster
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## Costs

GPT-4o-mini pricing:

- ~$0.15/1M input tokens
- ~$0.60/1M output tokens

GPT-4o pricing:

- ~$2.50/1M input tokens
- ~$10/1M output tokens

## Troubleshooting

### Issue: API key not found

Check that you ran the setup script or set the environment variable correctly.

### Issue: Package not installed

```bash
pip install openai
```

### Issue: Rate limit exceeded

Wait and retry, or check OpenAI dashboard for usage.

### Issue: Setup script not working

Make sure you're in the project directory:

```bash
cd Project_Clustery
python setup_api_keys.py
```

## Security

- Never commit API keys to git
- The .env file is automatically added to .gitignore
- Monitor API usage in OpenAI dashboard
