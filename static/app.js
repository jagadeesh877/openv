/**
 * OpenEnv Dashboard Core Logic
 */

class OpenEnvApp {
    constructor() {
        this.currentTask = 'email_triage';
        this.state = null;
        this.history = [];
        
        // DOM Elements
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
            evaluationStatus: document.getElementById('evaluation-status')
        };

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.syncState();
        
        // Start polling for state updates (for agent runs)
        setInterval(() => this.syncState(true), 2000);
    }

    setupEventListeners() {
        this.els.taskCards.forEach(card => {
            card.addEventListener('click', () => {
                const newTask = card.dataset.task;
                if (newTask !== this.currentTask) {
                    this.switchTask(newTask);
                    this.els.taskCards.forEach(c => c.classList.remove('active'));
                    card.classList.add('active');
                }
            });
        });

        this.els.btnReset.addEventListener('click', () => this.resetEnvironment());
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
            const data = await resp.json();
            await this.updateUI(data.observation);
            this.addFeedItem(actionType, data.reward);
        } catch (err) {
            console.error('Action failed:', err);
            this.notify('Action Error', 'Could not execute step.', 'error');
        }
    }

    async updateUI(data) {
        if (!data) return;
        this.state = data;

        // Determine if we're looking at a full state or just an observation
        const obs = data.observation || data;

        // Update Stats (Safe access)
        const taskId = obs.task_id || "unknown";
        this.els.statTask.textContent = taskId.replace(/_/g, ' ').toUpperCase();
        this.els.statStep.textContent = `${obs.step || 0} / 30`;
        
        // Update Description
        this.els.stateDescription.textContent = obs.state_description || "No state description available.";

        // Update Actions
        this.renderActions(obs.available_actions);

        // Update Context
        this.renderContext(obs.context);

        // Update Visualizer
        this.renderVisualization(obs);

        // Update Score (Real-time fallback logic)
        // Show final_score if the task is done, otherwise show cumulative_reward as a metric of progress
        const rawScore = data.final_score !== undefined ? data.final_score : 0;
        const reward = obs.cumulative_reward !== undefined ? obs.cumulative_reward : (data.reward ? data.reward.cumulative : 0);
        
        const displayScore = obs.done ? rawScore : reward;
        this.els.finalScore.textContent = displayScore.toFixed(2);
        
        if (obs.done) {
            this.els.evaluationStatus.textContent = rawScore >= 0.5 ? "✅ EVALUATION PASSED" : "❌ EVALUATION FAILED";
            this.els.evaluationStatus.style.color = rawScore >= 0.5 ? "var(--accent-green)" : "var(--accent-red)";
        } else {
            this.els.evaluationStatus.textContent = "Awaiting task completion";
            this.els.evaluationStatus.style.color = "var(--text-secondary)";
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
            btn.onclick = () => this.performAction(action);
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
