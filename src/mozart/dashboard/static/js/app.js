/* Mozart Dashboard — Alpine.js Application */

document.addEventListener('alpine:init', () => {
    Alpine.data('mozartApp', () => ({
        darkMode: localStorage.getItem('darkMode') === 'true' ||
                  (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches),
        mobileMenu: false,
        notifications: [],
        _notifId: 0,

        init() {
            // Apply dark mode class on init
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            }

            // Watch dark mode changes
            this.$watch('darkMode', (val) => {
                document.documentElement.classList.toggle('dark', val);
                localStorage.setItem('darkMode', val);
            });
        },

        addNotification(message, type = 'info') {
            const id = ++this._notifId;
            this.notifications.push({ id, message, type });
            // Auto-dismiss after 5 seconds
            setTimeout(() => this.removeNotification(id), 5000);
        },

        removeNotification(id) {
            this.notifications = this.notifications.filter(n => n.id !== id);
        }
    }));
});

/* htmx error handling */
document.addEventListener('htmx:responseError', (event) => {
    const status = event.detail.xhr?.status;
    const msg = status === 0
        ? 'Network error — is the server running?'
        : `Request failed (${status})`;

    const app = document.querySelector('[x-data]')?.__x;
    if (app && app.$data?.addNotification) {
        app.$data.addNotification(msg, 'error');
    }
});

/* htmx connection error */
document.addEventListener('htmx:sendError', () => {
    const app = document.querySelector('[x-data]')?.__x;
    if (app && app.$data?.addNotification) {
        app.$data.addNotification('Connection lost — retrying...', 'warning');
    }
});

/* Health check response handler — update conductor dot */
document.addEventListener('htmx:afterRequest', (event) => {
    if (event.detail.pathInfo?.requestPath === '/health') {
        const dot = document.getElementById('conductor-dot');
        if (!dot) return;
        if (event.detail.successful) {
            dot.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse-dot';
        } else {
            dot.className = 'w-2 h-2 rounded-full bg-red-500';
        }
        // Prevent htmx from swapping the health JSON into the dot
        event.detail.shouldSwap = false;
    }
});
