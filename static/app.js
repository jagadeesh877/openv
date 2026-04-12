/**
 * OpenEnv Dashboard Core Logic
 */

class OpenEnvApp {
    constructor() {
        // 1. Define elements first
        this.els = {
            taskCards: document.querySelectorAll('.task-card'),
            btnReset: document.getElementById('btn-reset'),
            statTask: document.querySelector('#stat-task .stat-value'),
            statStep: document.querySelector('#stat-step .stat-value'),
            statReward: document.querySelector('#stat-reward .stat-value'),
            stateDescription: document.getElementById('state-description'),
            actionButtons: document.getElementById('action-buttons'),
            observationContext: document.getElementById('observation-context'),
            agentFeed: document.getElementById('agent-feed'),
            visualizerPanel: document.getElementById('visualizer-panel'),
            finalScore: document.getElementById('final-score'),
            statScore: document.querySelector('#stat-score .stat-value'),
            btnBenchmark: document.getElementById('btn-run-benchmark'),
            btnRunLLM: document.getElementById('btn-run-llm'),
            toggleAutoPilot: document.getElementById('toggle-autopilot'),
            agentStatusBar: document.getElementById('agent-status-bar'),
            agentStatusText: document.getElementById('agent-status-text'),
            agentProgressBar: document.getElementById('agent-progress-bar'),
            // New Scoreboard Elements
            evalBoard: document.getElementById('evaluation-board'),
            evalTableBody: document.getElementById('eval-table-body'),
            evalLiveIndicator: document.getElementById('eval-live-indicator')
        };

        // 2. Setup error handler
        window.onerror = (msg, url, line, col, error) => {
            this.notify('Browser Error', `${msg} (line ${line})`, 'error');
            console.error('JS Error:', error);
            return false;
        };

        this.currentTask = 'email_triage';
        this.state = null;
        this.history = [];
        this.autoPilot = false;
        this.isBenchmarking = false;

        this.validateElements();
        this.init();
    }

    validateElements() {
        const missing = Object.entries(this.els)
            .filter(([key, val]) => !val)
            .map(([key]) => key);
        
        if (missing.length > 0) {
            console.warn('Dashboard: Some elements are missing from the DOM:', missing);
        } else {
            console.log('Dashboard: All UI elements resolved successfully.');
        }
    }

    async init() {
        this.setupEventListeners();
        await this.syncState();
        
        // Start polling for state updates (for agent runs)
        setInterval(() => this.syncState(true), 2000);
    }

    setupEventListeners() {
        if (this.els.taskCards) {
            this.els.taskCards.forEach(card => {
                card.addEventListener('click', (e) => {
                    const target = e.currentTarget;
                    const newTask = target.dataset.task;
                    if (newTask !== this.currentTask) {
                        this.switchTask(newTask);
                        this.els.taskCards.forEach(c => c.classList.remove('active'));
                        target.classList.add('active');
                    }
                });
            });
        }

        this.els.btnReset?.addEventListener('click', (e) => {
            const btn = e.currentTarget;
            btn.style.transform = 'scale(0.95)';
            setTimeout(() => { if (btn) btn.style.transform = 'scale(1)'; }, 100);
            console.log('Reset button clicked');
            this.resetEnvironment();
        });
        
        this.els.btnBenchmark?.addEventListener('click', (e) => {
            const btn = e.currentTarget;
            btn.style.transform = 'scale(0.95)';
            setTimeout(() => { if (btn) btn.style.transform = 'scale(1)'; }, 100);
            console.log('Benchmark button clicked');
            this.runFullBenchmark();
        });
        
        this.els.btnRunLLM?.addEventListener('click', (e) => {
            const btn = e.currentTarget;
            btn.style.transform = 'scale(0.95)';
            setTimeout(() => { if (btn) btn.style.transform = 'scale(1)'; }, 100);
            console.log('LLM Eval button clicked');
            this.runLLMEvaluation();
        });
        
        this.els.toggleAutoPilot?.addEventListener('change', (e) => {
            console.log('Auto-Pilot toggled:', e.target.checked);
            this.autoPilot = e.target.checked;
            if (this.autoPilot) this.runAutoPilotCycle();
        });
    }

    async switchTask(taskId) {
        this.currentTask = taskId;
        await this.resetEnvironment();
    }

    async resetEnvironment() {
        try {
            const resp = await fetch('/reset', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: this.currentTask })
            });
            const data = await resp.json();
            this.history = [];
            this.els.agentFeed.innerHTML = '';
            await this.updateUI(data);
            this.notify('Environment Reset', `Task '${this.currentTask}' initialized.`);
        } catch (err) {
            console.error('Reset failed:', err);
            this.notify('Reset Failed', 'Check server logs.', 'error');
        }
    }

    async syncState(quiet = false) {
        try {
            const resp = await fetch('/state');
            const data = await resp.json();
            
            // Only update if something changed or it's the first load
            if (!this.state || this.state.step !== data.step || this.state.task_id !== data.task_id) {
                await this.updateUI(data.observation || data);
            }
        } catch (err) {
            console.error('State sync failed:', err);
        }
    }

    async performAction(actionType, params = {}) {
        try {
            const resp = await fetch('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action_type: actionType, parameters: params })
            });
            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || 'Step failed');
            }
            const data = await resp.json();
            await this.updateUI(data.observation);
            this.addFeedItem(actionType, data.reward);
            
            // If auto-pilot is on, trigger next cycle
            if (this.autoPilot && !data.done) {
                setTimeout(() => this.runAutoPilotCycle(), 800);
            }
        } catch (err) {
            console.error('Action failed:', err);
            this.notify('Action Error', 'Could not execute step.', 'error');
            this.els.toggleAutoPilot.checked = false;
            this.autoPilot = false;
        }
    }

    async runAutoPilotCycle() {
        if (!this.autoPilot || (this.state && this.state.done)) return;

        try {
            const resp = await fetch('/baseline-suggestion');
            if (!resp.ok) throw new Error('Suggestion failed');
            const data = await resp.json();
            
            if (data.action === 'finish') {
                this.autoPilot = false;
                this.els.toggleAutoPilot.checked = false;
                return;
            }
            
            await this.performAction(data.action, data.parameters);
        } catch (err) {
            console.error('Auto-pilot cycle failed:', err);
            this.autoPilot = false;
            this.els.toggleAutoPilot.checked = false;
        }
    }

    async runLLMEvaluation() {
        if (this.isBenchmarking) return;
        
        this.isBenchmarking = true;
        this.els.btnRunLLM.disabled = true;
        this.els.btnRunLLM.textContent = '⏳ Evaluating...';
        this.els.agentStatusBar.classList.remove('hidden');
        
        this.notify('LLM Evaluation Started', 'The agent will solve all tasks using Llama-3.1.');

        try {
            const resp = await fetch('/run-llm-benchmark', { method: 'POST' });
            const data = await resp.json();
            
            if (data.status === 'already_running') {
                this.notify('System Busy', 'An evaluation is already in progress.', 'warning');
                // Still start polling in case it's actually running
                this.pollInferenceStatus();
                return;
            }
            
            if (!resp.ok) throw new Error('Failed to start LLM benchmark');
            
            // Start polling status
            this.pollInferenceStatus();
        } catch (err) {
            console.error('LLM Benchmark failed:', err);
            this.notify('Evaluation Error', 'Check server logs.', 'error');
            this.isBenchmarking = false;
            this.els.btnRunLLM.disabled = false;
            this.els.btnRunLLM.textContent = '🚀 Run LLM Evaluation';
            this.els.agentStatusBar.classList.add('hidden');
        }
    }

    async pollInferenceStatus() {
        if (this.pollTimer) clearInterval(this.pollTimer);
        
        // Show scoreboard, activate 'Live' indicator
        if (this.els.evalLiveIndicator) this.els.evalLiveIndicator.classList.remove('hidden');
        
        this.pollTimer = setInterval(async () => {
            try {
                const resp = await fetch('/inference-status');
                const data = await resp.json();
                
                if (!data.is_running) {
                    clearInterval(this.pollTimer);
                    this.pollTimer = null;
                    if (this.els.evalLiveIndicator) this.els.evalLiveIndicator.classList.add('hidden');
                    await this.finishEvaluation();
                    return;
                }
                
                this.els.agentStatusText.textContent = data.status_message || data.current_status || "Thinking...";
                
                // Live Scoreboard: Highlight active row and update partials
                if (data.current_task) {
                    this.updateScoreboardRow(data.current_task, data);
                }
                
                // Watch the agent work in real-time
                await this.syncState(true);
            } catch (err) {
                console.error('Status polling error:', err);
            }
        }, 1000);
    }

    updateScoreboardRow(taskId, data) {
        const row = this.els.evalTableBody.querySelector(`tr[data-task="${taskId}"]`);
        if (!row) return;

        // Clear other highlights
        this.els.evalTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('active-task'));
        row.classList.add('active-task');

        // Update step count from live state
        if (this.state && this.state.task_id === taskId) {
            row.querySelector('.eval-step').textContent = this.state.step || 0;
            const score = (this.state.current_score || 0).toFixed(2);
            row.querySelector('.eval-score').textContent = score;
        }
    }

    async finishEvaluation() {
        this.resetBenchmarkingState();
        if (this.els.evalLiveIndicator) this.els.evalLiveIndicator.classList.add('hidden');
        
        try {
            const resp = await fetch('/benchmark-results');
            if (!resp.ok) throw new Error('No results available');
            const data = await resp.json();
            
            // Final Board Update
            Object.entries(data.results).forEach(([taskId, result]) => {
                const row = this.els.evalTableBody.querySelector(`tr[data-task="${taskId}"]`);
                if (row) {
                    row.querySelector('.eval-step').textContent = result.steps;
                    row.querySelector('.eval-score').textContent = (result.final_score || 0).toFixed(4);
                    row.classList.remove('active-task');
                }
            });

            const mean = (Object.values(data.results).reduce((acc, r) => acc + (r.final_score || 0), 0) / 3).toFixed(4);
            this.els.finalScore.textContent = mean;
            
            this.notify('Evaluation Complete', `Scoreboard finalized: ${mean}`, 'success');
            await this.syncState(true);
        } catch (err) {
            console.warn('Final scoreboard update failed:', err);
        }
    }

    resetBenchmarkingState() {
        this.isBenchmarking = false;
        this.els.btnBenchmark.disabled = false;
        this.els.btnBenchmark.textContent = '🏆 Run Baseline';
        this.els.btnRunLLM.disabled = false;
        this.els.btnRunLLM.textContent = '🚀 Run LLM Evaluation';
        this.els.agentStatusBar.classList.add('hidden');
    }

    async runFullBenchmark() {
        if (this.isBenchmarking) return;
        
        this.isBenchmarking = true;
        this.els.btnBenchmark.disabled = true;
        this.els.btnBenchmark.textContent = '⏳ Calculating...';
        this.els.agentStatusBar.classList.remove('hidden');
        this.els.agentStatusText.textContent = "Running baseline agent...";

        try {
            const resp = await fetch('/run-baseline');
            if (!resp.ok) throw new Error('Benchmark failed');
            
            // Poll status just like LLM evaluation
            this.pollInferenceStatus();
        } catch (err) {
            console.error('Benchmark failed:', err);
            this.notify('Benchmark Error', 'Check server connectivity.', 'error');
            this.resetBenchmarkingState();
        }
    }

    showBenchmarkReport(data) {
        if (!data || !data.results) {
            this.notify('Benchmark Error', 'Invalid data returned from server.', 'error');
            return;
        }

        // Create a result table overlay
        const overlay = document.createElement('div');
        overlay.className = 'benchmark-overlay';
        
        let tableHtml = `
            <div class="result-modal">
                <header>
                    <h2>Benchmark Evaluation Result</h2>
                    <p>Agent: ${data.baseline_agent || 'unknown'} | ${data.timestamp || new Date().toISOString()}</p>
                </header>
                <div class="table-scroll">
                    <table class="result-table">
                        <thead>
                            <tr>
                                <th>Task</th>
                                <th>Steps</th>
                                <th>Rewards</th>
                                <th>Final Score</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        Object.entries(data.results).forEach(([task, res]) => {
            if (!res) return;
            const score = res.final_score || 0;
            const reward = res.total_reward || 0;
            const steps = res.steps || 0;
            const passed = score >= 0.5;
            
            tableHtml += `
                <tr>
                    <td>${task.replace(/_/g, ' ').toUpperCase()}</td>
                    <td>${steps}</td>
                    <td>${reward.toFixed(2)}</td>
                    <td class="${passed ? 'text-success' : 'text-error'}">${score.toFixed(4)}</td>
                    <td>${passed ? '✅ PASS' : '❌ FAIL'}</td>
                </tr>
            `;
        });
        
        tableHtml += `
                        </tbody>
                    </table>
                </div>
                <div class="modal-footer">
                    <div class="avg-score">MEAN BENCHMARK SCORE: <span>${(data.average_score || 0).toFixed(4)}</span></div>
                    <button class="btn btn-primary" onclick="this.closest('.benchmark-overlay').remove()">Close Results</button>
                </div>
            </div>
        `;
        
        overlay.innerHTML = tableHtml;
        document.body.appendChild(overlay);
    }

    async updateUI(data) {
        if (!data) return;
        try {
            this.state = data.observation || data;
            
            // Update Stats
            if (this.els.statTask) this.els.statTask.textContent = (data.task_id || '').replace(/_/g, ' ').toUpperCase();
            if (this.els.statStep) this.els.statStep.textContent = `${this.state.step || 0} / 30`;
            if (this.els.statReward) this.els.statReward.textContent = (this.state.cumulative_reward || 0).toFixed(4);
            
            const perc = Math.round((this.state.current_score || 0) * 100);
            if (this.els.statScore) this.els.statScore.textContent = `${perc}%`;
            
            if (this.els.stateDescription) this.els.stateDescription.textContent = this.state.state_description || "No description";
            
            this.renderActions(this.state.available_actions || []);
            this.renderContext(this.state.context || {});
            
            // Critical: Update the central workspace view
            this.renderVisualization(this.state);
            
            // Heartbeat check
            this.updateSystemStatus("Online");
        } catch (err) {
            console.error('Update UI error:', err);
        }
    }

    updateSystemStatus(status) {
        const ping = document.getElementById('ping-status');
        if (ping) {
            ping.textContent = `Server: ${status}`;
            ping.className = status === 'Online' ? 'status-online' : 'status-offline';
        }
    }

    renderActions(actions) {
        this.els.actionButtons.innerHTML = '';
        if (!actions || actions.length === 0) {
            this.els.actionButtons.innerHTML = '<p class="empty-hint">No actions available in this state.</p>';
            return;
        }

        actions.forEach(action => {
            const btn = document.createElement('button');
            btn.className = 'btn-action';
            btn.textContent = action;
            if (this.state && this.state.done) {
                btn.disabled = true;
                btn.title = "Task completed. Reset to play again.";
            } else {
                btn.onclick = () => this.performAction(action);
            }
            this.els.actionButtons.appendChild(btn);
        });
    }

    renderContext(context) {
        this.els.observationContext.innerHTML = '';
        if (!context) return;

        Object.entries(context).forEach(([key, val]) => {
            if (typeof val === 'object') val = JSON.stringify(val);
            const div = document.createElement('div');
            div.className = 'context-row';
            div.innerHTML = `<span class="context-key">${key}:</span> <span class="context-value">${val}</span>`;
            this.els.observationContext.appendChild(div);
        });
    }

    renderVisualization(obs) {
        const taskId = obs.task_id;
        const ctx = obs.context || {};
        const panel = this.els.visualizerPanel;
        panel.innerHTML = '';

        const container = document.createElement('div');
        container.className = 'visualization-box';

        if (taskId === 'email_triage') {
            const emails = ctx.emails || [];
            const list = document.createElement('div');
            list.className = 'email-list';
            emails.forEach((email, idx) => {
                const item = document.createElement('div');
                item.className = `email-item ${idx === ctx.current_index ? 'active' : ''}`;
                item.innerHTML = `<strong>From:</strong> ${email.sender}<br><strong>Subject:</strong> ${email.subject}<br><small>${email.content.substring(0, 60)}...</small>`;
                list.appendChild(item);
            });
            container.appendChild(list);
        } else if (taskId === 'code_review') {
            const snippet = ctx.current_snippet || "// No snippet loaded";
            const codeBox = document.createElement('pre');
            codeBox.className = 'code-box';
            codeBox.textContent = snippet;
            container.appendChild(codeBox);
        } else if (taskId === 'meeting_scheduler') {
            const schedule = ctx.schedule || {};
            const grid = document.createElement('div');
            grid.className = 'calendar-grid';
            // Placeholder 9-5 schedule mapping
            for (let h = 9; h <= 17; h++) {
                const hour = `${h.toString().padStart(2, '0')}00`;
                const slot = document.createElement('div');
                const isOccupied = schedule[hour];
                slot.className = `slot ${isOccupied ? 'occupied' : ''}`;
                slot.textContent = `${hour}: ${isOccupied || 'FREE'}`;
                grid.appendChild(slot);
            }
            container.appendChild(grid);
        }

        panel.appendChild(container);
    }

    addFeedItem(action, reward) {
        if (!reward) return;
        
        // Update reward stat
        this.els.statReward.textContent = reward.cumulative.toFixed(4);
        
        const item = document.createElement('div');
        const isPos = reward.value > 0;
        const isNeg = reward.value < 0;
        item.className = `feed-item ${isPos ? 'reward-positive' : ''} ${isNeg ? 'reward-negative' : ''}`;
        
        item.innerHTML = `
            <div class="feed-meta">
                <span>Step ${this.state.step}</span>
                <span>${isPos ? '+' : ''}${reward.value.toFixed(2)}</span>
            </div>
            <div class="feed-title">${action}</div>
            <div class="feed-reason">${reward.reason}</div>
        `;
        
        this.els.agentFeed.prepend(item);
    }

    async fetchFinalScore() {
        try {
            const resp = await fetch('/state');
            const data = await resp.json();
            const score = data.final_score || 0;
            this.els.finalScore.textContent = score.toFixed(2);
            this.els.evaluationStatus.textContent = score >= 0.5 ? "✅ EVALUATION PASSED" : "❌ EVALUATION FAILED";
            this.els.evaluationStatus.style.color = score >= 0.5 ? "var(--accent-green)" : "var(--accent-red)";
        } catch (err) {
            console.error('Core dump failed');
        }
    }

    notify(title, msg, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) {
            console.warn('Notification container missing from DOM');
            return;
        }
        const note = document.createElement('div');
        note.className = `notification ${type}`;
        note.innerHTML = `<strong>${title}</strong><p>${msg}</p>`;
        container.appendChild(note);
        setTimeout(() => note.remove(), 4000);
    }
}

// Global Notification Styles (Added to DOM dynamically)
const style = document.createElement('style');
style.textContent = `
    .notification-container { position: fixed; top: 20px; right: 20px; z-index: 1000; display: flex; flex-direction: column; gap: 10px; }
    .notification { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border-color); border-radius: 12px; padding: 12px 20px; box-shadow: var(--shadow-soft); animation: slideIn 0.3s ease-out; min-width: 250px; }
    .notification.error { border-left: 4px solid var(--accent-red); }
`;
document.head.appendChild(style);

document.addEventListener('DOMContentLoaded', () => {
    window.app = new OpenEnvApp();
});
