/**
 * Local AI Chatbot - Frontend Logic & SSE Streaming
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentSessionId = localStorage.getItem('active_session_id') || createNewSessionId();
  let sessions = JSON.parse(localStorage.getItem('chat_sessions') || '[]');
  let isGenerating = false;
  let abortController = null;
  let pendingUpload = null;

  // Settings State
  const savedProvider = localStorage.getItem('cfg_provider');
  const initialProvider = (savedProvider && savedProvider !== 'mock') ? savedProvider : 'groq';
  const savedModel = localStorage.getItem('cfg_model');
  const initialModel = (savedModel && savedModel !== 'demo-assistant') ? savedModel : 'openai/gpt-oss-120b';

  let config = {
    provider: initialProvider,
    model: initialModel,
    temperature: parseFloat(localStorage.getItem('cfg_temp') || '0.7'),
    systemPrompt: localStorage.getItem('cfg_system_prompt') || '',
  };

  // DOM Elements
  const messagesContainer = document.getElementById('chat-messages');
  const welcomeHero = document.getElementById('welcome-hero');
  const userInput = document.getElementById('user-input');
  const chatForm = document.getElementById('chat-form');
  const sendBtn = document.getElementById('send-btn');
  const newChatBtn = document.getElementById('new-chat-btn');
  const sessionsList = document.getElementById('sessions-list');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const exportChatBtn = document.getElementById('export-chat-btn');
  const uploadDocLabel = document.getElementById('upload-doc-label');
  const ragFileInput = document.getElementById('rag-file-input');
  const attachmentPreview = document.getElementById('attachment-preview');
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const themeIcon = document.getElementById('theme-icon');
  const themeText = document.getElementById('theme-text');
  const sidebar = document.getElementById('sidebar');
  const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');

  // Header and Settings Elements
  const activeModelPill = document.getElementById('active-model-pill');
  const headerProviderName = document.getElementById('header-provider-name');
  const headerModelName = document.getElementById('header-model-name');
  const settingsModal = document.getElementById('settings-modal');
  const openSettingsBtn = document.getElementById('open-settings-btn');
  const closeSettingsBtn = document.getElementById('close-settings-btn');
  const saveSettingsBtn = document.getElementById('save-settings-btn');
  const providerSelect = document.getElementById('provider-select');
  const providerDesc = document.getElementById('provider-desc');
  const modelSelect = document.getElementById('model-select');
  const customModelGroup = document.getElementById('custom-model-group');
  const customModelInput = document.getElementById('custom-model-input');
  const tempSlider = document.getElementById('temp-slider');
  const tempVal = document.getElementById('temp-val');
  const systemPromptInput = document.getElementById('system-prompt');

  // Provider Default Model Mapping & Descriptions
  const providerDefaults = {
    groq: {
      desc: 'Groq ultra-fast AI acceleration.',
      models: [
        { id: 'openai/gpt-oss-120b', label: 'openai/gpt-oss-120b (Recommended)' },
        { id: 'qwen/qwen3.8-27b', label: 'qwen/qwen3.8-27b' },
        { id: 'qwen/qwen3.6-27b', label: 'qwen/qwen3.6-27b' },
        { id: 'openai/gpt-oss-20b', label: 'openai/gpt-oss-20b' },
        { id: 'groq/compound', label: 'groq/compound' },
      ],    },
    ollama: {
      desc: '100% Free local models running via Ollama on your computer.',
      models: [
        { id: 'llama3.2', label: 'llama3.2 (Default)' },
        { id: 'deepseek-r1:latest', label: 'deepseek-r1:latest' },
        { id: 'mistral', label: 'mistral' },
        { id: 'qwen2.5', label: 'qwen2.5' },
        { id: 'phi3', label: 'phi3' },
      ],
    },
    openai: {
      desc: 'OpenAI GPT-4o models (requires OPENAI_API_KEY in .env).',
      models: [
        { id: 'gpt-4o-mini', label: 'gpt-4o-mini (Fast & Cost-effective)' },
        { id: 'gpt-4o', label: 'gpt-4o (High Intelligence)' },
        { id: 'gpt-3.5-turbo', label: 'gpt-3.5-turbo' },
      ],
    },
    gemini: {
      desc: 'Google Gemini Flash / Pro (requires GEMINI_API_KEY in .env).',
      models: [
        { id: 'gemini-1.5-flash', label: 'gemini-1.5-flash (Fast Default)' },
        { id: 'gemini-1.5-pro', label: 'gemini-1.5-pro (High Reasoning)' },
        { id: 'gemini-2.0-flash-exp', label: 'gemini-2.0-flash-exp' },
      ],
    },
    anthropic: {
      desc: 'Anthropic Claude models (requires ANTHROPIC_API_KEY in .env).',
      models: [
        { id: 'claude-3-5-sonnet-20241022', label: 'claude-3-5-sonnet-20241022 (Recommended)' },
        { id: 'claude-3-5-haiku-20241022', label: 'claude-3-5-haiku-20241022 (Fast)' },
      ],
    },
    openrouter: {
      desc: 'Unified multi-provider API (requires OPENROUTER_API_KEY in .env).',
      models: [
        { id: 'poolside/laguna-s-2.1:free', label: 'poolside/laguna-s-2.1:free' },
        { id: 'meta-llama/llama-3.3-70b-instruct:free', label: 'meta-llama/llama-3.3-70b-instruct:free' },
        { id: 'deepseek/deepseek-r1:free', label: 'deepseek/deepseek-r1:free' },
      ],
    },
  };

  // Helper to populate model dropdown for a given provider
  function populateModelOptions(provider, selectedModel) {
    if (!modelSelect) return;
    modelSelect.innerHTML = '';
    const provInfo = providerDefaults[provider] || { models: [] };
    const modelList = provInfo.models || [];

    let isKnownModel = false;
    modelList.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.label || m.id;
      if (selectedModel && selectedModel === m.id) {
        opt.selected = true;
        isKnownModel = true;
      }
      modelSelect.appendChild(opt);
    });

    // Option for entering a custom model tag
    const customOpt = document.createElement('option');
    customOpt.value = '__custom__';
    customOpt.textContent = 'Custom model (type manually)...';
    modelSelect.appendChild(customOpt);

    if (selectedModel && !isKnownModel) {
      customOpt.selected = true;
      if (customModelGroup) customModelGroup.style.display = 'flex';
      if (customModelInput) customModelInput.value = selectedModel;
    } else {
      if (!selectedModel && modelList.length > 0) {
        modelSelect.value = modelList[0].id;
      }
      if (customModelGroup) customModelGroup.style.display = 'none';
      if (customModelInput) customModelInput.value = '';
    }
  }

  // Fetch backend /api/models to dynamically sync any server-configured models
  async function syncServerModels() {
    try {
      const res = await fetch('/api/models');
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.providers) {
        for (const [key, val] of Object.entries(data.providers)) {
          if (!providerDefaults[key]) {
            providerDefaults[key] = { desc: val.description || '', models: [] };
          }
          if (Array.isArray(val.models) && val.models.length > 0) {
            providerDefaults[key].models = val.models.map((m) => {
              if (typeof m === 'string') {
                return { id: m, label: m };
              }
              return m;
            });
          }
          if (val.description) {
            providerDefaults[key].desc = val.description;
          }
        }
      }
    } catch (e) {
      console.warn('Could not sync models from /api/models:', e);
    }
  }

  // Initialize
  initTheme();
  fetchSessions();
  loadSessionHistory(currentSessionId);
  setupAutoResize();

  // Sync models from server, then refresh the header pill with up-to-date info
  syncServerModels().then(() => {
    updateHeaderPill();
  });

  // Event Listeners
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  newChatBtn.addEventListener('click', () => {
    startNewChat();
  });

  clearChatBtn.addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear this conversation?')) {
      await fetch(`/api/history/${currentSessionId}`, { method: 'DELETE' });
      messagesContainer.innerHTML = '';
      messagesContainer.appendChild(welcomeHero);
      welcomeHero.style.display = 'flex';
    }
  });

  exportChatBtn.addEventListener('click', exportConversation);

  // Document Upload (RAG)
  if (uploadDocLabel && ragFileInput && attachmentPreview) {
    ragFileInput.addEventListener('change', async () => {
      const file = ragFileInput.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      // Show uploading chip
      attachmentPreview.style.display = 'flex';
      attachmentPreview.innerHTML = `
        <div id="upload-chip" style="display: flex; align-items: center; gap: 8px; background: var(--bg-input); border: 1px solid var(--border-color); padding: 6px 12px; border-radius: var(--radius-sm); font-size: 0.85rem;">
          <span style="font-size: 1.2rem;">📄</span>
          <span id="upload-chip-text" style="color: var(--text-primary); font-weight: 500;">${escapeHtml(file.name)} (Uploading...)</span>
        </div>
      `;

      const originalHtml = uploadDocLabel.innerHTML;
      uploadDocLabel.innerHTML = '⏳';
      uploadDocLabel.style.pointerEvents = 'none';

      welcomeHero.style.display = 'none';
      pendingUpload = null;

      /**
       * Poll GET /api/rag/documents/{documentId} every 1.5 s until the
       * status field is "ingested" or "failed", then update the chip.
       */
      async function pollIngestStatus(documentId, filename) {
        const chipText = document.getElementById('upload-chip-text');
        const chip = document.getElementById('upload-chip');
        if (chipText) chipText.textContent = `${escapeHtml(filename)} (Processing…)`;

        const MAX_WAIT_MS = 5 * 60 * 1000; // 5-minute safety ceiling
        const INTERVAL_MS = 1500;
        const started = Date.now();

        while (Date.now() - started < MAX_WAIT_MS) {
          await new Promise((r) => setTimeout(r, INTERVAL_MS));

          let poll;
          try {
            const res = await fetch(`/api/rag/documents/${documentId}`);
            if (!res.ok) continue; // transient error — keep polling
            poll = await res.json();
          } catch {
            continue; // network blip — keep polling
          }

          const status = poll.status;

          if (status === 'ingested') {
            const chunks = poll.chunk_count ?? 0;
            if (chipText) chipText.textContent = `${escapeHtml(filename)} (${chunks} chunks ready)`;
            if (chip) chip.style.borderColor = 'var(--accent-color)';

            pendingUpload = {
              name: filename,
              chunks,
              html: `
                <div style="display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,0.05); border: 1px solid var(--border-color); padding: 10px 14px; border-radius: var(--radius-md); margin-bottom: 12px; width: fit-content;">
                  <span style="font-size: 1.5rem;">📄</span>
                  <div>
                    <div style="font-weight: 600; font-size: 0.9rem;">${escapeHtml(filename)}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Indexed (${chunks} chunks)</div>
                  </div>
                </div>
              `,
            };
            return;
          }

          if (status === 'failed') {
            const errMsg = poll.error_message || 'Unknown error';
            if (chipText) chipText.textContent = `${escapeHtml(filename)} (Error: ${escapeHtml(errMsg)})`;
            if (chip) chip.style.borderColor = '#ef4444';
            return;
          }
          // status is still "processing" — loop and wait
        }

        // Timed out
        if (document.getElementById('upload-chip-text'))
          document.getElementById('upload-chip-text').textContent = `${escapeHtml(filename)} (Timed out — check back later)`;
        if (document.getElementById('upload-chip'))
          document.getElementById('upload-chip').style.borderColor = '#f59e0b';
      }

      try {
        const resp = await fetch('/api/rag/ingest', {
          method: 'POST',
          body: formData,
        });

        // Safe-parse: some error responses (e.g. unhandled 500s) are plain text.
        let data;
        const contentType = resp.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          data = await resp.json();
        } else {
          const text = await resp.text();
          // Try to parse anyway in case the Content-Type header is wrong.
          try { data = JSON.parse(text); }
          catch { data = { detail: text || `HTTP ${resp.status}` }; }
        }

        if (!resp.ok) {
          throw new Error(data.detail?.message || data.detail || `Server error ${resp.status}`);
        }

        if (data.status === 'duplicate') {
          document.getElementById('upload-chip-text').textContent = `${escapeHtml(file.name)} (Already ingested)`;
          document.getElementById('upload-chip').style.borderColor = 'var(--accent-color)';

          pendingUpload = {
            name: file.name,
            chunks: 0,
            html: `
              <div style="display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,0.05); border: 1px solid var(--border-color); padding: 10px 14px; border-radius: var(--radius-md); margin-bottom: 12px; width: fit-content;">
                <span style="font-size: 1.5rem;">📄</span>
                <div>
                  <div style="font-weight: 600; font-size: 0.9rem;">${escapeHtml(file.name)}</div>
                  <div style="font-size: 0.75rem; color: var(--text-muted);">Already in knowledge base</div>
                </div>
              </div>
            `,
          };
        } else if (data.status === 'processing') {
          // 202 — kick off background poll; UI updates asynchronously
          pollIngestStatus(data.document_id, file.name);
        }
      } catch (err) {
        const chipEl = document.getElementById('upload-chip-text');
        const chipWrap = document.getElementById('upload-chip');
        if (chipEl) chipEl.textContent = `${escapeHtml(file.name)} (Error: ${escapeHtml(err.message)})`;
        if (chipWrap) chipWrap.style.borderColor = '#ef4444';
      } finally {
        uploadDocLabel.innerHTML = originalHtml;
        uploadDocLabel.style.pointerEvents = 'auto';
        ragFileInput.value = '';
      }
    });
  }


  // Suggestion chips
  document.querySelectorAll('.suggestion-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      userInput.value = prompt;
      handleSendMessage();
    });
  });

  // Settings Modal Events
  activeModelPill.addEventListener('click', openSettings);
  openSettingsBtn.addEventListener('click', openSettings);
  closeSettingsBtn.addEventListener('click', closeSettings);
  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) closeSettings();
  });

  providerSelect.addEventListener('change', () => {
    const selected = providerSelect.value;
    if (providerDefaults[selected]) {
      providerDesc.textContent = providerDefaults[selected].desc || '';
      populateModelOptions(selected);
    }
  });

  modelSelect.addEventListener('change', () => {
    if (modelSelect.value === '__custom__') {
      customModelGroup.style.display = 'flex';
      customModelInput.focus();
    } else {
      customModelGroup.style.display = 'none';
    }
  });

  tempSlider.addEventListener('input', () => {
    tempVal.textContent = tempSlider.value;
  });

  saveSettingsBtn.addEventListener('click', () => {
    config.provider = providerSelect.value;
    
    let chosenModel = modelSelect.value;
    if (chosenModel === '__custom__') {
      chosenModel = customModelInput.value.trim();
    }
    const defaultModel = providerDefaults[config.provider]?.models?.[0]?.id || 'gpt-4o-mini';
    config.model = chosenModel || defaultModel;
    config.temperature = parseFloat(tempSlider.value);
    config.systemPrompt = systemPromptInput.value.trim();

    localStorage.setItem('cfg_provider', config.provider);
    localStorage.setItem('cfg_model', config.model);
    localStorage.setItem('cfg_temp', config.temperature.toString());
    localStorage.setItem('cfg_system_prompt', config.systemPrompt);

    updateHeaderPill();
    closeSettings();
  });

  // Theme Toggle
  themeToggleBtn.addEventListener('click', toggleTheme);

  // Mobile sidebar toggle
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }

  // -------------------------------------------------------------
  // Messaging & Streaming Engine
  // -------------------------------------------------------------
  async function handleSendMessage() {
    const text = userInput.value.trim();
    if ((!text && !pendingUpload) || isGenerating) return;

    // Reset textarea
    userInput.value = '';
    userInput.style.height = 'auto';

    const attachmentHtml = pendingUpload ? pendingUpload.html : '';
    const uploadedFileName = pendingUpload ? pendingUpload.name : '';
    if (pendingUpload) {
      // Clear preview
      attachmentPreview.innerHTML = '';
      attachmentPreview.style.display = 'none';
      pendingUpload = null;
    }

    // Hide welcome hero if visible
    if (welcomeHero && welcomeHero.parentElement) {
      welcomeHero.style.display = 'none';
    }

    // Save session title if new
    ensureSessionExists(text);

    // Append User Message to UI
    appendMessageUI('user', text, attachmentHtml);

    // Prepare for backend
    let backendMessage = text;
    if (uploadedFileName) {
      backendMessage = `[System context: The user has just uploaded a document into the knowledge base named "${uploadedFileName}". If their message is short or ambiguous like "what is this" or "summarize", they are referring to this document. Please use the retrieve_documents tool to look it up.]\n\nUser message: ${text}`;
    } else if (!text && uploadedFileName) {
      backendMessage = `[System context: The user uploaded a document named "${uploadedFileName}" without any accompanying message. Please use the retrieve_documents tool to search for it and summarize it for the user.]`;
    }

    // Prepare Assistant Message Placeholder with live cursor
    const { bubbleEl, toolContainerEl, rowEl, sourcesContainerEl } = createAssistantMessageUI();
    isGenerating = true;
    sendBtn.disabled = true;

    let fullAssistantResponse = '';
    abortController = new AbortController();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: backendMessage,
          session_id: currentSessionId,
          provider: config.provider,
          model: config.model,
          system_prompt: config.systemPrompt || undefined,
          temperature: config.temperature,
        }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            if (!dataStr) continue;

            try {
              const event = JSON.parse(dataStr);
              if (event.type === 'token') {
                fullAssistantResponse += event.content;
                renderAssistantMarkdown(bubbleEl, fullAssistantResponse, true);
                scrollToBottom();
              } else if (event.type === 'tool_start') {
                renderToolBadge(toolContainerEl, event.name, 'running', event.args);
              } else if (event.type === 'tool_end') {
                renderToolBadge(toolContainerEl, event.name, 'done', null, event.result);
              } else if (event.type === 'error') {
                fullAssistantResponse += `\n\n⚠️ **Error:** ${event.content}`;
                renderAssistantMarkdown(bubbleEl, fullAssistantResponse, false);
              } else if (event.type === 'citations') {
                if (event.citations && event.citations.length > 0) {
                  renderCitations(sourcesContainerEl, event.citations);
                  scrollToBottom();
                }
              } else if (event.type === 'done') {
                // finished
              }
            } catch (err) {
              console.error('SSE JSON parse error:', err, dataStr);
            }
          }
        }
      }

      // Finalize Markdown formatting
      renderAssistantMarkdown(bubbleEl, fullAssistantResponse, false);
      highlightCodeBlocks(bubbleEl);
    } catch (err) {
      if (err.name !== 'AbortError') {
        renderAssistantMarkdown(bubbleEl, `⚠️ **Connection Error:** ${err.message}`, false);
      }
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      scrollToBottom();
    }
  }

  // -------------------------------------------------------------
  // UI Rendering Helpers
  // -------------------------------------------------------------
  function appendMessageUI(role, content, attachmentHtml = '') {
    const row = document.createElement('div');
    row.className = `message-row ${role === 'user' ? 'user-row' : 'bot-row'}`;

    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${role === 'user' ? 'user-avatar' : 'bot-avatar'}`;
    avatar.textContent = role === 'user' ? '👤' : '⚡';

    const wrapper = document.createElement('div');
    wrapper.className = 'message-content-wrapper';

    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${role === 'user' ? 'user-bubble' : 'bot-bubble'}`;

    if (role === 'user') {
      bubble.innerHTML = attachmentHtml + escapeHtml(content).replace(/\\n/g, '<br>');
    } else {
      bubble.innerHTML = DOMPurify.sanitize(marked.parse(content));
      highlightCodeBlocks(bubble);
    }

    wrapper.appendChild(bubble);
    if (role === 'user') {
      row.appendChild(wrapper);
      row.appendChild(avatar);
    } else {
      row.appendChild(avatar);
      row.appendChild(wrapper);
    }

    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function createAssistantMessageUI() {
    const row = document.createElement('div');
    row.className = 'message-row bot-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.textContent = '⚡';

    const wrapper = document.createElement('div');
    wrapper.className = 'message-content-wrapper';

    const toolContainer = document.createElement('div');
    toolContainer.className = 'tool-badges-list';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-bubble';
    bubble.innerHTML = '<span class="typing-cursor"></span>';

    // Placeholder appended after the bubble — renderCitations() will populate it.
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'sources-section';
    sourcesContainer.style.display = 'none';

    wrapper.appendChild(toolContainer);
    wrapper.appendChild(bubble);
    wrapper.appendChild(sourcesContainer);
    row.appendChild(avatar);
    row.appendChild(wrapper);

    messagesContainer.appendChild(row);
    scrollToBottom();

    return { bubbleEl: bubble, toolContainerEl: toolContainer, rowEl: row, sourcesContainerEl: sourcesContainer };
  }

  function renderAssistantMarkdown(bubbleEl, text, isTyping) {
    if (!text && isTyping) {
      bubbleEl.innerHTML = '<span class="typing-cursor"></span>';
      return;
    }
    const html = DOMPurify.sanitize(marked.parse(text));
    bubbleEl.innerHTML = html + (isTyping ? '<span class="typing-cursor"></span>' : '');
  }

  function renderToolBadge(container, toolName, status, args, result) {
    let badge = container.querySelector(`[data-tool="${toolName}"]`);
    if (!badge) {
      badge = document.createElement('div');
      badge.className = 'tool-badge';
      badge.setAttribute('data-tool', toolName);
      container.appendChild(badge);
    }

    if (status === 'running') {
      badge.innerHTML = `<span class="tool-spinner"></span> <span>Running <strong>${toolName}</strong>...</span>`;
    } else {
      badge.innerHTML = `<span>✓</span> <span>Used <strong>${toolName}</strong></span>`;
    }
  }

  /**
   * Render a "Sources" section below the assistant bubble.
   *
   * Each citation becomes a pill; clicking it toggles a collapsible panel
   * showing the retrieved snippet text plus provenance metadata.
   *
   * @param {HTMLElement} container  - The .sources-section div created in createAssistantMessageUI.
   * @param {Array}       citations  - Array of citation objects from the SSE "citations" event.
   *   Each object has: index, filename, page_number, section, chunk_id,
   *                    dense_score, fusion_score, rerank_score.
   *   The snippet text is carried separately in the ToolMessage content (not
   *   re-sent in the citation event), so the panel shows provenance only.
   */
  function renderCitations(container, citations) {
    if (!container || !citations || citations.length === 0) return;

    container.innerHTML = '';
    container.style.display = 'flex';

    // Label row
    const label = document.createElement('div');
    label.className = 'sources-label';
    label.innerHTML = '<span>📚</span> Sources';
    container.appendChild(label);

    // Pills row
    const pillsRow = document.createElement('div');
    pillsRow.className = 'sources-pills';
    container.appendChild(pillsRow);

    // One snippet wrapper per citation (hidden by default)
    const snippetWrappers = [];

    citations.forEach((c, i) => {
      // Build pill label: "filename (p. N)" or just "filename"
      const pageLabel = c.page_number ? ` · p. ${c.page_number}` : '';
      const sectionLabel = c.section ? ` · ${c.section}` : '';
      const pillText = `${escapeHtml(c.filename || 'Unknown')}${pageLabel}`;

      const pill = document.createElement('button');
      pill.className = 'source-pill';
      pill.setAttribute('aria-expanded', 'false');
      pill.innerHTML = `<span class="pill-icon">📄</span>${pillText}`;

      // Snippet wrapper (CSS-animated expand/collapse)
      const snippetWrapper = document.createElement('div');
      snippetWrapper.className = 'source-snippet-wrapper';

      // Score display: prefer rerank > fusion > dense
      const score = c.rerank_score ?? c.fusion_score ?? c.dense_score;
      const scoreLabel = score != null ? `${(score * 100).toFixed(1)}% relevance` : '';

      const metaParts = [];
      if (c.page_number) metaParts.push(`<span>📄 Page ${c.page_number}</span>`);
      if (c.section)     metaParts.push(`<span>§ ${escapeHtml(c.section)}</span>`);
      if (scoreLabel)    metaParts.push(`<span>🎯 ${scoreLabel}</span>`);
      if (c.chunk_id)    metaParts.push(`<span style="opacity:0.5">id: ${c.chunk_id}</span>`);

      snippetWrapper.innerHTML = `
        <div class="source-snippet">
          <div class="source-snippet-meta">${metaParts.join('') || '&nbsp;'}</div>
        </div>
      `;

      // Toggle logic — only one snippet open at a time within this message
      pill.addEventListener('click', () => {
        const isOpen = snippetWrapper.classList.contains('open');

        // Close all siblings first
        snippetWrappers.forEach((sw, j) => {
          sw.classList.remove('open');
          pillsRow.querySelectorAll('.source-pill')[j]?.classList.remove('active');
          pillsRow.querySelectorAll('.source-pill')[j]?.setAttribute('aria-expanded', 'false');
        });

        if (!isOpen) {
          snippetWrapper.classList.add('open');
          pill.classList.add('active');
          pill.setAttribute('aria-expanded', 'true');
        }
        scrollToBottom();
      });

      snippetWrappers.push(snippetWrapper);
      pillsRow.appendChild(pill);
      container.appendChild(snippetWrapper);
    });
  }

  function highlightCodeBlocks(el) {
    el.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
      // Add copy button if not exists
      const pre = block.parentElement;
      if (!pre.querySelector('.code-header')) {
        const lang = block.className.replace('language-', '') || 'code';
        const header = document.createElement('div');
        header.className = 'code-header';
        header.innerHTML = `
          <span>${lang}</span>
          <button class="copy-code-btn">Copy</button>
        `;
        const copyBtn = header.querySelector('.copy-code-btn');
        copyBtn.addEventListener('click', () => {
          navigator.clipboard.writeText(block.innerText).then(() => {
            copyBtn.textContent = 'Copied!';
            setTimeout(() => (copyBtn.textContent = 'Copy'), 2000);
          });
        });
        pre.insertBefore(header, block);
      }
    });
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // -------------------------------------------------------------
  // Session Management (Synced with Database)
  // -------------------------------------------------------------
  function createNewSessionId() {
    const id = 'sess_' + Math.random().toString(36).substring(2, 9);
    localStorage.setItem('active_session_id', id);
    return id;
  }

  async function fetchSessions() {
    try {
      const res = await fetch('/api/sessions');
      if (res.ok) {
        sessions = await res.json();
        localStorage.setItem('chat_sessions', JSON.stringify(sessions));
      }
    } catch (e) {
      console.warn('Could not fetch sessions from DB:', e);
    }
    renderSessions();
  }

  async function startNewChat() {
    currentSessionId = createNewSessionId();
    try {
      await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: currentSessionId, title: 'New Chat' }),
      });
    } catch (e) {
      console.warn(e);
    }
    messagesContainer.innerHTML = '';
    messagesContainer.appendChild(welcomeHero);
    welcomeHero.style.display = 'flex';
    await fetchSessions();
  }

  async function ensureSessionExists(firstMessage) {
    let session = sessions.find((s) => s.id === currentSessionId);
    if (!session) {
      const title = firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '...' : '');
      try {
        await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: currentSessionId, title: title }),
        });
      } catch (e) {
        console.warn(e);
      }
      await fetchSessions();
    }
  }

  function renderSessions() {
    sessionsList.innerHTML = '';
    sessions.forEach((s) => {
      const item = document.createElement('div');
      item.className = `session-item ${s.id === currentSessionId ? 'active' : ''}`;
      item.innerHTML = `
        <span class="session-title">${escapeHtml(s.title || 'New Chat')}</span>
        <button class="session-del-btn" title="Delete conversation">&times;</button>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('session-del-btn')) {
          e.stopPropagation();
          deleteSession(s.id);
          return;
        }
        switchSession(s.id);
      });

      sessionsList.appendChild(item);
    });
  }

  async function switchSession(id) {
    if (id === currentSessionId) return;
    currentSessionId = id;
    localStorage.setItem('active_session_id', id);
    renderSessions();
    await loadSessionHistory(id);
    if (sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  }

  async function deleteSession(id) {
    sessions = sessions.filter((s) => s.id !== id);
    localStorage.setItem('chat_sessions', JSON.stringify(sessions));
    await fetch(`/api/sessions/${id}`, { method: 'DELETE' });

    if (id === currentSessionId) {
      startNewChat();
    } else {
      renderSessions();
    }
  }

  async function loadSessionHistory(sessionId) {
    try {
      const res = await fetch(`/api/history/${sessionId}`);
      const data = await res.json();
      messagesContainer.innerHTML = '';

      if (!data.messages || data.messages.length === 0) {
        messagesContainer.appendChild(welcomeHero);
        welcomeHero.style.display = 'flex';
      } else {
        welcomeHero.style.display = 'none';
        data.messages.forEach((msg) => {
          appendMessageUI(msg.role, msg.content);
        });
      }
    } catch (err) {
      console.warn('Could not load history:', err);
    }
  }

  function exportConversation() {
    fetch(`/api/history/${currentSessionId}`)
      .then((res) => res.json())
      .then((data) => {
        let text = `# Chat Export - Session ${currentSessionId}\n\n`;
        (data.messages || []).forEach((m) => {
          text += `### ${m.role === 'user' ? 'User' : 'Assistant'}:\n${m.content}\n\n---\n\n`;
        });
        const blob = new Blob([text], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_${currentSessionId}.md`;
        a.click();
      });
  }

  // -------------------------------------------------------------
  // Settings & Theme
  // -------------------------------------------------------------
  function openSettings() {
    providerSelect.value = config.provider;
    populateModelOptions(config.provider, config.model);
    tempSlider.value = config.temperature;
    tempVal.textContent = config.temperature;
    systemPromptInput.value = config.systemPrompt;
    providerDesc.textContent = providerDefaults[config.provider]?.desc || '';
    settingsModal.classList.add('open');
  }

  function closeSettings() {
    settingsModal.classList.remove('open');
  }

  function updateHeaderPill() {
    const opt = Array.from(providerSelect.options).find(o => o.value === config.provider);
    headerProviderName.textContent = opt ? opt.text.split(' ')[0] : config.provider.toUpperCase();
    headerModelName.textContent = config.model;
  }

  function initTheme() {
    const savedTheme = localStorage.getItem('app_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeUI(savedTheme);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('app_theme', next);
    updateThemeUI(next);
  }

  // -------------------------------------------------------------
  // Long-Term Memory (Mem0) Modal Management
  // -------------------------------------------------------------
  const memoriesModal = document.getElementById('memories-modal');
  const openMemoriesBtn = document.getElementById('open-memories-btn');
  const closeMemoriesBtn = document.getElementById('close-memories-btn');
  const closeMemoriesDoneBtn = document.getElementById('close-memories-done-btn');
  const clearAllMemoriesBtn = document.getElementById('clear-all-memories-btn');
  const memoriesList = document.getElementById('memories-list');

  openMemoriesBtn.addEventListener('click', openMemories);
  closeMemoriesBtn.addEventListener('click', closeMemories);
  closeMemoriesDoneBtn.addEventListener('click', closeMemories);
  memoriesModal.addEventListener('click', (e) => {
    if (e.target === memoriesModal) closeMemories();
  });

  clearAllMemoriesBtn.addEventListener('click', async () => {
    if (confirm('Clear all long-term memories learned about you?')) {
      await fetch('/api/memories/default_user', { method: 'DELETE' });
      await loadMemories();
    }
  });

  async function openMemories() {
    memoriesModal.classList.add('open');
    await loadMemories();
  }

  function closeMemories() {
    memoriesModal.classList.remove('open');
  }

  async function loadMemories() {
    memoriesList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">Loading memories...</div>';
    try {
      const res = await fetch('/api/memories/default_user');
      const data = await res.json();
      memoriesList.innerHTML = '';
      if (!data.memories || data.memories.length === 0) {
        memoriesList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem; padding: 12px; text-align: center; border: 1px dashed var(--border-color); border-radius: var(--radius-sm);">No facts stored yet. Start chatting and the AI will remember details about you!</div>';
        return;
      }

      data.memories.forEach((m) => {
        const item = document.createElement('div');
        item.style.cssText = 'display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-sm); font-size: 0.88rem;';
        item.innerHTML = `
          <span>${escapeHtml(m.memory || m)}</span>
          <button class="mem-del-btn" style="background: none; border: none; color: var(--text-muted); cursor: pointer; padding: 2px 6px; font-size: 1rem;" title="Delete this memory">&times;</button>
        `;
        const delBtn = item.querySelector('.mem-del-btn');
        delBtn.addEventListener('click', async () => {
          if (m.id) {
            await fetch(`/api/memories/default_user/${m.id}`, { method: 'DELETE' });
          }
          await loadMemories();
        });
        memoriesList.appendChild(item);
      });
    } catch (e) {
      memoriesList.innerHTML = '<div style="color: #ef4444;">Failed to load memories.</div>';
    }
  }

  function updateThemeUI(theme) {
    if (theme === 'dark') {
      themeIcon.textContent = '🌙';
      themeText.textContent = 'Dark Mode';
    } else {
      themeIcon.textContent = '☀️';
      themeText.textContent = 'Light Mode';
    }
  }

  function setupAutoResize() {
    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
    });
  }

  function escapeHtml(str) {
    return (str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
