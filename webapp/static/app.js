const TelegramWebApp = window.Telegram?.WebApp;
const DEBUG = true;

function debugLog(msg) {
    const timestamp = new Date().toLocaleTimeString('id-ID', { hour12: false });
    const fullMsg = `[${timestamp}] ${msg}`;
    console.log('[APP] ' + fullMsg);
    const debugEl = document.getElementById('debug-console');
    if (debugEl && DEBUG) {
        debugEl.classList.add('active');
        debugEl.innerHTML += fullMsg + '<br>';
        if (debugEl.childElementCount > 50) {
            debugEl.removeChild(debugEl.firstChild);
        }
        debugEl.scrollTop = debugEl.scrollHeight;
    }
}

let isConnected = false;
let lastUpdateTime = null;
let updateInterval = null;
let fetchInProgress = false;

let chart = null;
let candleSeries = null;
let entryLine = null;
let slLine = null;
let tpLine = null;
let currentPriceLine = null;
let lastActivePosition = null;

debugLog('app.js loaded - v2.0');

function initTelegram() {
    debugLog('initTelegram called');
    if (TelegramWebApp) {
        try {
            TelegramWebApp.ready();
            TelegramWebApp.expand();
            debugLog('Telegram WebApp initialized');
            
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
        } catch (e) {
            debugLog('Telegram init error: ' + e.message);
        }
    } else {
        debugLog('Telegram WebApp not available - running standalone');
    }
}

async function fetchWithRetry(url, maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            debugLog(`Fetch attempt ${attempt}/${maxRetries}: ${url}`);
            const urlWithCache = url.includes('?') ? url + '&t=' + Date.now() : url + '?t=' + Date.now();
            const response = await fetch(urlWithCache, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                },
                cache: 'no-store'
            });
            
            debugLog(`Response status: ${response.status}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const text = await response.text();
            debugLog(`Response length: ${text.length} chars`);
            
            if (!text || text.trim() === '') {
                throw new Error('Empty response body');
            }
            
            try {
                const data = JSON.parse(text);
                debugLog(`✅ Fetch OK: ${url} - data keys: ${Object.keys(data).join(', ')}`);
                return data;
            } catch (parseError) {
                debugLog(`❌ JSON parse error: ${parseError.message}`);
                debugLog(`Raw response (first 200 chars): ${text.substring(0, 200)}`);
                throw new Error(`JSON parse failed: ${parseError.message}`);
            }
        } catch (error) {
            debugLog(`❌ Attempt ${attempt}/${maxRetries} failed: ${error.message}`);
            if (attempt === maxRetries) {
                throw error;
            }
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
    }
}

async function fetchDashboardData() {
    debugLog('fetchDashboardData called');
    const data = await fetchWithRetry('/api/dashboard');
    if (data) {
        debugLog(`Dashboard data received: price=${data.price ? 'yes' : 'no'}, signal=${data.last_signal ? 'yes' : 'no'}, position=${data.active_position ? 'yes' : 'no'}`);
    }
    return data;
}

async function fetchCandlesData() {
    debugLog('fetchCandlesData called');
    const data = await fetchWithRetry('/api/candles?timeframe=M1&limit=100');
    if (data) {
        debugLog(`Candles data received: ${data.candles ? data.candles.length : 0} candles`);
    }
    return data;
}

function convertToWIB(timestamp) {
    try {
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) {
            debugLog(`Invalid timestamp: ${timestamp}`);
            return Math.floor(Date.now() / 1000);
        }
        const wibOffset = 7 * 60 * 60 * 1000;
        const utcTime = date.getTime() + (date.getTimezoneOffset() * 60 * 1000);
        const wibTime = new Date(utcTime + wibOffset);
        return Math.floor(wibTime.getTime() / 1000);
    } catch (e) {
        debugLog(`convertToWIB error: ${e.message}`);
        return Math.floor(Date.now() / 1000);
    }
}

function initChart() {
    const container = document.getElementById('chart-container');
    debugLog('initChart called. Container exists: ' + (!!container) + ', LWC available: ' + (typeof LightweightCharts !== 'undefined'));
    
    if (!container) {
        debugLog('ERROR: Chart container not found');
        return false;
    }
    
    if (typeof LightweightCharts === 'undefined') {
        debugLog('ERROR: LightweightCharts not loaded');
        return false;
    }
    
    try {
        if (chart) {
            chart.remove();
            chart = null;
            candleSeries = null;
        }
        
        chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 300,
            layout: {
                background: { type: 'solid', color: '#1a1a2e' },
                textColor: '#ffffff'
            },
            grid: {
                vertLines: { color: '#2a2a4e' },
                horzLines: { color: '#2a2a4e' }
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal
            },
            rightPriceScale: {
                borderColor: '#2a2a4e',
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.1
                }
            },
            timeScale: {
                borderColor: '#2a2a4e',
                timeVisible: true,
                secondsVisible: false
            },
            localization: {
                timeFormatter: (timestamp) => {
                    const date = new Date(timestamp * 1000);
                    return date.toLocaleTimeString('id-ID', {
                        hour: '2-digit',
                        minute: '2-digit',
                        timeZone: 'Asia/Jakarta'
                    });
                }
            }
        });
        
        candleSeries = chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderUpColor: '#22c55e',
            borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444'
        });
        
        window.addEventListener('resize', () => {
            if (chart && container) {
                chart.applyOptions({ width: container.clientWidth });
            }
        });
        
        debugLog('Chart initialized successfully');
        return true;
    } catch (e) {
        debugLog(`Chart init error: ${e.message}`);
        return false;
    }
}

function removePriceLines() {
    if (candleSeries) {
        try {
            if (entryLine) {
                candleSeries.removePriceLine(entryLine);
                entryLine = null;
            }
            if (slLine) {
                candleSeries.removePriceLine(slLine);
                slLine = null;
            }
            if (tpLine) {
                candleSeries.removePriceLine(tpLine);
                tpLine = null;
            }
            if (currentPriceLine) {
                candleSeries.removePriceLine(currentPriceLine);
                currentPriceLine = null;
            }
        } catch (e) {
            debugLog(`removePriceLines error: ${e.message}`);
        }
    }
}

function updatePriceLines(position, currentPrice) {
    removePriceLines();
    
    if (!candleSeries) return;
    
    try {
        if (currentPrice && currentPrice > 0) {
            currentPriceLine = candleSeries.createPriceLine({
                price: currentPrice,
                color: '#5eaeff',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                axisLabelVisible: true,
                title: 'NOW'
            });
        }
        
        if (!position) return;
        
        const direction = (position.direction || 'BUY').toUpperCase();
        const entryColor = direction === 'BUY' ? '#22c55e' : '#ef4444';
        
        if (position.entry_price) {
            entryLine = candleSeries.createPriceLine({
                price: position.entry_price,
                color: entryColor,
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                axisLabelVisible: true,
                title: direction === 'BUY' ? '📈 ENTRY' : '📉 ENTRY'
            });
        }
        
        if (position.stop_loss) {
            slLine = candleSeries.createPriceLine({
                price: position.stop_loss,
                color: '#ef4444',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: '🛑 SL'
            });
        }
        
        if (position.take_profit) {
            tpLine = candleSeries.createPriceLine({
                price: position.take_profit,
                color: '#ffd700',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: '🎯 TP'
            });
        }
        
        if (chart && position.entry_price) {
            const watermarkText = direction === 'BUY' ? '📈 LONG POSITION ACTIVE' : '📉 SHORT POSITION ACTIVE';
            chart.applyOptions({
                watermark: {
                    visible: true,
                    fontSize: 14,
                    horzAlign: 'center',
                    vertAlign: 'top',
                    color: 'rgba(255, 255, 255, 0.3)',
                    text: watermarkText
                }
            });
        }
    } catch (e) {
        debugLog(`updatePriceLines error: ${e.message}`);
    }
}

function clearChartWatermark() {
    if (chart) {
        try {
            chart.applyOptions({
                watermark: {
                    visible: false
                }
            });
        } catch (e) {
            debugLog(`clearChartWatermark error: ${e.message}`);
        }
    }
}

async function updateCandleChart() {
    debugLog('updateCandleChart called');
    try {
        const data = await fetchCandlesData();
        
        if (!data) {
            debugLog('No candles data returned');
            return;
        }
        
        if (!candleSeries) {
            debugLog('candleSeries not initialized');
            return;
        }
        
        if (!data.candles || data.candles.length === 0) {
            debugLog('No candles in data');
            return;
        }
        
        debugLog(`Processing ${data.candles.length} candles`);
        
        const chartData = data.candles.map(candle => ({
            time: convertToWIB(candle.timestamp),
            open: parseFloat(candle.open),
            high: parseFloat(candle.high),
            low: parseFloat(candle.low),
            close: parseFloat(candle.close)
        })).filter(candle => 
            !isNaN(candle.time) && 
            !isNaN(candle.open) && 
            !isNaN(candle.high) && 
            !isNaN(candle.low) && 
            !isNaN(candle.close)
        );
        
        chartData.sort((a, b) => a.time - b.time);
        
        const uniqueData = [];
        const seenTimes = new Set();
        for (const candle of chartData) {
            if (!seenTimes.has(candle.time)) {
                seenTimes.add(candle.time);
                uniqueData.push(candle);
            }
        }
        
        debugLog(`Setting ${uniqueData.length} unique candles to chart`);
        candleSeries.setData(uniqueData);
        
        const currentPrice = data.current_price || (uniqueData.length > 0 ? uniqueData[uniqueData.length - 1].close : null);
        
        if (data.active_position) {
            lastActivePosition = data.active_position;
            updatePriceLines(data.active_position, currentPrice);
        } else {
            if (lastActivePosition) {
                clearChartWatermark();
            }
            lastActivePosition = null;
            removePriceLines();
            if (currentPrice) {
                updatePriceLines(null, currentPrice);
            }
        }
        
        debugLog('Chart updated successfully');
        
    } catch (error) {
        debugLog(`Error updating candle chart: ${error.message}`);
        console.error('Error updating candle chart:', error);
    }
}

function formatPrice(price) {
    if (price === null || price === undefined || isNaN(price)) return '--';
    return parseFloat(price).toFixed(2);
}

function formatPercent(value) {
    if (value === null || value === undefined || isNaN(value)) return '--';
    const num = parseFloat(value);
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}%`;
}

function formatCurrency(value) {
    if (value === null || value === undefined || isNaN(value)) return '--';
    const num = parseFloat(value);
    const sign = num >= 0 ? '+' : '';
    return `${sign}$${Math.abs(num).toFixed(2)}`;
}

function formatTime(timestamp) {
    if (!timestamp) return '--';
    try {
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) return '--';
        return date.toLocaleTimeString('id-ID', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit',
            hour12: false 
        });
    } catch (e) {
        return '--';
    }
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
    debugLog('updatePriceCard called');
    
    const priceElement = document.getElementById('price-main');
    const changeElement = document.getElementById('price-change');
    const bidElement = document.getElementById('price-bid');
    const askElement = document.getElementById('price-ask');
    const spreadElement = document.getElementById('price-spread');
    const rangeElement = document.getElementById('price-range');
    
    if (!data) {
        debugLog('updatePriceCard: no data');
        return;
    }
    
    if (!data.price) {
        debugLog('updatePriceCard: no price data in response');
        return;
    }
    
    debugLog(`Price data: mid=${data.price.mid}, bid=${data.price.bid}, ask=${data.price.ask}`);
    
    if (priceElement) {
        const midPrice = data.price.mid || data.price.quote || ((data.price.bid + data.price.ask) / 2);
        priceElement.textContent = formatPrice(midPrice);
        debugLog(`Updated price-main: ${formatPrice(midPrice)}`);
    }
    
    if (changeElement) {
        const changePercent = data.price.change_percent || 0;
        changeElement.textContent = formatPercent(changePercent);
        changeElement.className = 'price-change ' + (changePercent >= 0 ? 'positive' : 'negative');
    }
    
    if (bidElement) {
        bidElement.textContent = formatPrice(data.price.bid);
    }
    
    if (askElement) {
        askElement.textContent = formatPrice(data.price.ask);
    }
    
    if (spreadElement) {
        const spread = data.price.spread || (data.price.ask && data.price.bid ? (data.price.ask - data.price.bid) * 10 : 0);
        spreadElement.textContent = (spread || 0).toFixed(1) + ' pips';
    }
    
    if (rangeElement && data.price.high && data.price.low) {
        rangeElement.textContent = formatPrice(data.price.low) + ' - ' + formatPrice(data.price.high);
    }
}

function updateSignalCard(data) {
    debugLog('updateSignalCard called');
    
    const signalBadge = document.getElementById('signal-badge');
    const entryElement = document.getElementById('signal-entry');
    const slElement = document.getElementById('signal-sl');
    const tpElement = document.getElementById('signal-tp');
    const timeElement = document.getElementById('signal-time');
    
    if (!signalBadge) {
        debugLog('updateSignalCard: signal-badge element not found');
        return;
    }
    
    if (data && data.last_signal && data.last_signal.direction) {
        const direction = data.last_signal.direction.toUpperCase();
        const icon = direction === 'BUY' ? '📈' : '📉';
        signalBadge.innerHTML = `${icon} ${direction}`;
        signalBadge.className = 'signal-badge ' + direction.toLowerCase();
        
        if (entryElement) entryElement.textContent = formatPrice(data.last_signal.entry_price);
        if (slElement) slElement.textContent = formatPrice(data.last_signal.sl);
        if (tpElement) tpElement.textContent = formatPrice(data.last_signal.tp);
        if (timeElement) timeElement.textContent = formatTime(data.last_signal.timestamp);
        
        debugLog(`Signal updated: ${direction} @ ${data.last_signal.entry_price}`);
    } else {
        signalBadge.innerHTML = '⏳ Menunggu Sinyal';
        signalBadge.className = 'signal-badge neutral';
        
        if (entryElement) entryElement.textContent = '--';
        if (slElement) slElement.textContent = '--';
        if (tpElement) tpElement.textContent = '--';
        if (timeElement) timeElement.textContent = '--';
        
        debugLog('No signal data available');
    }
}

function updatePositionCard(data) {
    debugLog('updatePositionCard called');
    
    const container = document.getElementById('position-container');
    
    if (!container) {
        debugLog('updatePositionCard: position-container not found');
        return;
    }
    
    if (data && data.active_position && data.active_position.active) {
        const pos = data.active_position;
        const direction = (pos.direction || pos.signal_type || 'BUY').toUpperCase();
        const pnl = pos.unrealized_pnl || pos.pnl || 0;
        const isProfit = pnl >= 0;
        
        const pnlPips = pos.current_pnl_pips || pos.pnl_pips || 0;
        const distanceToTP = pos.distance_to_tp_pips || '--';
        const distanceToSL = pos.distance_to_sl_pips || '--';
        
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
                    <div class="signal-info-value" style="color: #ef4444;">${formatPrice(pos.sl || pos.stop_loss)}</div>
                </div>
                <div class="signal-info-item">
                    <div class="signal-info-label">TP</div>
                    <div class="signal-info-value" style="color: #ffd700;">${formatPrice(pos.tp || pos.take_profit)}</div>
                </div>
            </div>
            <div class="position-details" style="margin-top: 8px;">
                <div class="signal-info-item">
                    <div class="signal-info-label">P/L Pips</div>
                    <div class="signal-info-value ${pnlPips >= 0 ? 'positive' : 'negative'}">${pnlPips >= 0 ? '+' : ''}${typeof pnlPips === 'number' ? pnlPips.toFixed(1) : pnlPips}</div>
                </div>
                <div class="signal-info-item">
                    <div class="signal-info-label">To TP</div>
                    <div class="signal-info-value" style="color: #ffd700;">${typeof distanceToTP === 'number' ? distanceToTP.toFixed(1) : distanceToTP} pips</div>
                </div>
                <div class="signal-info-item">
                    <div class="signal-info-label">To SL</div>
                    <div class="signal-info-value" style="color: #ef4444;">${typeof distanceToSL === 'number' ? distanceToSL.toFixed(1) : distanceToSL} pips</div>
                </div>
            </div>
        `;
        
        debugLog(`Position updated: ${direction} @ ${pos.entry_price}, PnL: ${pnl}`);
    } else {
        container.innerHTML = `
            <div class="no-position">
                <p>Tidak ada posisi aktif</p>
            </div>
        `;
        debugLog('No active position');
    }
}

function updateStatsCard(data) {
    debugLog('updateStatsCard called');
    
    const winRateElement = document.getElementById('stat-winrate');
    const totalPnlElement = document.getElementById('stat-pnl');
    const todaySignalsElement = document.getElementById('stat-signals');
    const tradesElement = document.getElementById('stat-trades');
    
    if (data && data.stats) {
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
        debugLog('Stats updated');
    } else {
        debugLog('No stats data available');
    }
}

function updateRegimeCard(data) {
    debugLog('updateRegimeCard called');
    
    const container = document.getElementById('regime-tags');
    
    if (!container) {
        debugLog('updateRegimeCard: regime-tags not found');
        return;
    }
    
    if (data && data.regime) {
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
        
        if (data.regime.confidence !== undefined && data.regime.confidence !== null) {
            const confPercent = (data.regime.confidence * 100).toFixed(0);
            tags.push(`<span class="regime-tag">Confidence: ${confPercent}%</span>`);
        }
        
        if (data.regime.type) {
            tags.push(`<span class="regime-tag">${data.regime.type}</span>`);
        }
        
        container.innerHTML = tags.length > 0 ? tags.join('') : '<span class="regime-tag">Data tidak tersedia</span>';
        debugLog('Regime updated');
    } else {
        container.innerHTML = '<span class="regime-tag">Data tidak tersedia</span>';
        debugLog('No regime data');
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
    if (fetchInProgress) {
        debugLog('Refresh skipped - fetch in progress');
        return;
    }
    
    fetchInProgress = true;
    debugLog('refreshData() started');
    
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.disabled = true;
    }
    
    try {
        debugLog('Fetching dashboard data...');
        const data = await fetchDashboardData();
        
        if (!data) {
            throw new Error('No data returned from API');
        }
        
        debugLog('Data received, updating UI...');
        
        updateConnectionStatus(true);
        updatePriceCard(data);
        updateSignalCard(data);
        updatePositionCard(data);
        updateStatsCard(data);
        updateRegimeCard(data);
        updateUpdateTime();
        
        debugLog('UI updates complete, updating chart...');
        await updateCandleChart();
        
        debugLog('refreshData() completed successfully');
        
    } catch (error) {
        debugLog(`ERROR in refreshData: ${error.message}`);
        console.error('Error refreshing data:', error);
        updateConnectionStatus(false);
        
        const priceElement = document.getElementById('price-main');
        if (priceElement && priceElement.textContent === '--') {
            priceElement.textContent = 'Memuat...';
        }
    } finally {
        fetchInProgress = false;
        if (refreshBtn) {
            refreshBtn.disabled = false;
        }
    }
}

function startAutoRefresh() {
    debugLog('startAutoRefresh called');
    
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    refreshData();
    
    updateInterval = setInterval(refreshData, 3000);
    debugLog('Auto-refresh started (interval: 3s)');
}

function stopAutoRefresh() {
    debugLog('stopAutoRefresh called');
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    debugLog('DOMContentLoaded fired');
    
    debugLog('Checking required elements...');
    const requiredElements = [
        'price-main', 'price-change', 'price-bid', 'price-ask', 
        'signal-badge', 'position-container', 'chart-container',
        'regime-tags', 'stat-winrate', 'stat-pnl'
    ];
    
    requiredElements.forEach(id => {
        const el = document.getElementById(id);
        debugLog(`Element #${id}: ${el ? 'found' : 'MISSING!'}`);
    });
    
    initTelegram();
    
    const chartInitialized = initChart();
    debugLog(`Chart init result: ${chartInitialized}`);
    
    startAutoRefresh();
    
    debugLog('Dashboard initialization complete');
    
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            debugLog('Manual refresh triggered');
            refreshData();
        });
    }
    
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            debugLog('Page hidden - stopping auto-refresh');
            stopAutoRefresh();
        } else {
            debugLog('Page visible - starting auto-refresh');
            startAutoRefresh();
        }
    });
});

window.addEventListener('beforeunload', () => {
    debugLog('Page unloading');
    stopAutoRefresh();
});

window.addEventListener('error', (event) => {
    debugLog(`Global error: ${event.message} at ${event.filename}:${event.lineno}`);
});

window.addEventListener('unhandledrejection', (event) => {
    debugLog(`Unhandled promise rejection: ${event.reason}`);
});
