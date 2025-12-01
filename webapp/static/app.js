const TelegramWebApp = window.Telegram?.WebApp;

let isConnected = false;
let lastUpdateTime = null;
let updateInterval = null;

function initTelegram() {
    if (TelegramWebApp) {
        TelegramWebApp.ready();
        TelegramWebApp.expand();
        
        const theme = TelegramWebApp.themeParams;
        if (theme.bg_color) {
            document.documentElement.style.setProperty('--tg-theme-bg-color', theme.bg_color);
        }
        if (theme.text_color) {
            document.documentElement.style.setProperty('--tg-theme-text-color', theme.text_color);
        }
        if (theme.hint_color) {
            document.documentElement.style.setProperty('--tg-theme-hint-color', theme.hint_color);
        }
        if (theme.button_color) {
            document.documentElement.style.setProperty('--tg-theme-button-color', theme.button_color);
        }
        if (theme.secondary_bg_color) {
            document.documentElement.style.setProperty('--tg-theme-secondary-bg-color', theme.secondary_bg_color);
        }
    }
}

async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
        throw error;
    }
}

function formatPrice(price) {
    if (price === null || price === undefined) return '--';
    return parseFloat(price).toFixed(2);
}

function formatPercent(value) {
    if (value === null || value === undefined) return '--';
    const num = parseFloat(value);
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}%`;
}

function formatCurrency(value) {
    if (value === null || value === undefined) return '--';
    const num = parseFloat(value);
    const sign = num >= 0 ? '+' : '';
    return `${sign}$${Math.abs(num).toFixed(2)}`;
}

function formatTime(timestamp) {
    if (!timestamp) return '--';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('id-ID', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit',
        hour12: false 
    });
}

function updateConnectionStatus(connected) {
    isConnected = connected;
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    if (statusDot && statusText) {
        if (connected) {
            statusDot.classList.remove('offline');
            statusText.textContent = 'LIVE';
        } else {
            statusDot.classList.add('offline');
            statusText.textContent = 'OFFLINE';
        }
    }
}

function updatePriceCard(data) {
    const priceElement = document.getElementById('price-main');
    const changeElement = document.getElementById('price-change');
    const bidElement = document.getElementById('price-bid');
    const askElement = document.getElementById('price-ask');
    const spreadElement = document.getElementById('price-spread');
    const rangeElement = document.getElementById('price-range');
    
    if (priceElement && data.price) {
        priceElement.textContent = formatPrice(data.price.mid);
        
        if (changeElement) {
            const changePercent = data.price.change_percent || 0;
            changeElement.textContent = formatPercent(changePercent);
            changeElement.className = 'price-change ' + (changePercent >= 0 ? 'positive' : 'negative');
        }
        
        if (bidElement) bidElement.textContent = formatPrice(data.price.bid);
        if (askElement) askElement.textContent = formatPrice(data.price.ask);
        if (spreadElement) spreadElement.textContent = (data.price.spread || 0).toFixed(1) + ' pips';
        if (rangeElement && data.price.high && data.price.low) {
            rangeElement.textContent = formatPrice(data.price.low) + ' - ' + formatPrice(data.price.high);
        }
    }
}

function updateSignalCard(data) {
    const signalBadge = document.getElementById('signal-badge');
    const entryElement = document.getElementById('signal-entry');
    const slElement = document.getElementById('signal-sl');
    const tpElement = document.getElementById('signal-tp');
    const timeElement = document.getElementById('signal-time');
    
    if (signalBadge) {
        if (data.last_signal && data.last_signal.direction) {
            const direction = data.last_signal.direction.toUpperCase();
            const icon = direction === 'BUY' ? '📈' : '📉';
            signalBadge.innerHTML = `${icon} ${direction}`;
            signalBadge.className = 'signal-badge ' + direction.toLowerCase();
            
            if (entryElement) entryElement.textContent = formatPrice(data.last_signal.entry_price);
            if (slElement) slElement.textContent = formatPrice(data.last_signal.sl);
            if (tpElement) tpElement.textContent = formatPrice(data.last_signal.tp);
            if (timeElement) timeElement.textContent = formatTime(data.last_signal.timestamp);
        } else {
            signalBadge.innerHTML = '⏳ Menunggu Sinyal';
            signalBadge.className = 'signal-badge neutral';
            
            if (entryElement) entryElement.textContent = '--';
            if (slElement) slElement.textContent = '--';
            if (tpElement) tpElement.textContent = '--';
            if (timeElement) timeElement.textContent = '--';
        }
    }
}

function updatePositionCard(data) {
    const container = document.getElementById('position-container');
    
    if (!container) return;
    
    if (data.active_position && data.active_position.active) {
        const pos = data.active_position;
        const direction = (pos.direction || 'BUY').toUpperCase();
        const pnl = pos.unrealized_pnl || 0;
        const isProfit = pnl >= 0;
        
        container.innerHTML = `
            <div class="position-header">
                <span class="position-type ${direction.toLowerCase()}">${direction === 'BUY' ? '📈' : '📉'} ${direction}</span>
                <span class="position-pnl ${isProfit ? 'positive' : 'negative'}">${formatCurrency(pnl)}</span>
            </div>
            <div class="position-details">
                <div class="signal-info-item">
                    <div class="signal-info-label">Entry</div>
                    <div class="signal-info-value">${formatPrice(pos.entry_price)}</div>
                </div>
                <div class="signal-info-item">
                    <div class="signal-info-label">SL</div>
                    <div class="signal-info-value">${formatPrice(pos.sl)}</div>
                </div>
                <div class="signal-info-item">
                    <div class="signal-info-label">TP</div>
                    <div class="signal-info-value">${formatPrice(pos.tp)}</div>
                </div>
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="no-position">
                <p>Tidak ada posisi aktif</p>
            </div>
        `;
    }
}

function updateStatsCard(data) {
    const winRateElement = document.getElementById('stat-winrate');
    const totalPnlElement = document.getElementById('stat-pnl');
    const todaySignalsElement = document.getElementById('stat-signals');
    const tradesElement = document.getElementById('stat-trades');
    
    if (data.stats) {
        if (winRateElement) {
            winRateElement.textContent = (data.stats.win_rate || 0).toFixed(1) + '%';
        }
        if (totalPnlElement) {
            const pnl = data.stats.total_pnl || 0;
            totalPnlElement.textContent = formatCurrency(pnl);
            totalPnlElement.className = 'stat-value ' + (pnl >= 0 ? 'positive' : 'negative');
        }
        if (todaySignalsElement) {
            todaySignalsElement.textContent = data.stats.signals_today || 0;
        }
        if (tradesElement) {
            tradesElement.textContent = data.stats.total_trades || 0;
        }
    }
}

function updateRegimeCard(data) {
    const container = document.getElementById('regime-tags');
    
    if (!container) return;
    
    if (data.regime) {
        let tags = [];
        
        if (data.regime.trend) {
            const trendClass = data.regime.trend.toLowerCase().includes('bullish') ? 'trend-up' : 
                              data.regime.trend.toLowerCase().includes('bearish') ? 'trend-down' : '';
            tags.push(`<span class="regime-tag ${trendClass}">${data.regime.trend}</span>`);
        }
        
        if (data.regime.volatility) {
            const volClass = data.regime.volatility.toLowerCase().includes('high') ? 'volatile' : '';
            tags.push(`<span class="regime-tag ${volClass}">${data.regime.volatility}</span>`);
        }
        
        if (data.regime.bias) {
            const biasClass = data.regime.bias === 'BUY' ? 'trend-up' : 
                             data.regime.bias === 'SELL' ? 'trend-down' : '';
            tags.push(`<span class="regime-tag ${biasClass}">Bias: ${data.regime.bias}</span>`);
        }
        
        if (data.regime.confidence) {
            tags.push(`<span class="regime-tag">Confidence: ${(data.regime.confidence * 100).toFixed(0)}%</span>`);
        }
        
        container.innerHTML = tags.length > 0 ? tags.join('') : '<span class="regime-tag">Data tidak tersedia</span>';
    } else {
        container.innerHTML = '<span class="regime-tag">Data tidak tersedia</span>';
    }
}

function updateUpdateTime() {
    const element = document.getElementById('update-time');
    if (element) {
        lastUpdateTime = new Date();
        element.textContent = 'Update terakhir: ' + formatTime(lastUpdateTime);
    }
}

async function refreshData() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.disabled = true;
    }
    
    try {
        const data = await fetchDashboardData();
        
        updateConnectionStatus(true);
        updatePriceCard(data);
        updateSignalCard(data);
        updatePositionCard(data);
        updateStatsCard(data);
        updateRegimeCard(data);
        updateUpdateTime();
        
    } catch (error) {
        console.error('Error refreshing data:', error);
        updateConnectionStatus(false);
    } finally {
        if (refreshBtn) {
            refreshBtn.disabled = false;
        }
    }
}

function startAutoRefresh() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    refreshData();
    
    updateInterval = setInterval(refreshData, 5000);
}

function stopAutoRefresh() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initTelegram();
    startAutoRefresh();
    
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            refreshData();
        });
    }
    
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopAutoRefresh();
        } else {
            startAutoRefresh();
        }
    });
});

window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
});
