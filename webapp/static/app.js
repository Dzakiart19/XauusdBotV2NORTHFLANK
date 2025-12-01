"use strict";

(function() {
    var DEBUG = (function() {
        try {
            var urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('debug') === 'true') return true;
            if (urlParams.get('debug') === 'false') return false;
        } catch (e) {}
        return false;
    })();
    
    var TelegramWebApp = window.Telegram ? window.Telegram.WebApp : null;

    function debugLog(msg) {
        var timestamp = new Date().toLocaleTimeString('id-ID', { hour12: false });
        var fullMsg = '[' + timestamp + '] ' + msg;
        if (DEBUG) {
            console.log('[APP] ' + fullMsg);
            var debugEl = document.getElementById('debug-console');
            if (debugEl) {
                debugEl.classList.add('active');
                debugEl.innerHTML += fullMsg + '<br>';
                debugEl.scrollTop = debugEl.scrollHeight;
            }
        }
    }

    window.debugLog = debugLog;
    debugLog('app.js loaded - v4.0 (Real-Time Dashboard)');

    var isConnected = false;
    var lastUpdateTime = null;
    var updateInterval = null;
    var chartUpdateInterval = null;
    var historyUpdateInterval = null;
    var fetchInProgress = false;
    var chart = null;
    var candleSeries = null;
    var entryLine = null;
    var slLine = null;
    var tpLine = null;
    var currentPriceLine = null;
    var lastActivePosition = null;
    var websocket = null;
    var wsReconnectTimer = null;
    var wsReconnectAttempts = 0;
    var MAX_WS_RECONNECT_ATTEMPTS = 15;
    var useWebSocket = true;
    var connectionState = 'disconnected';
    var lastPriceData = null;
    var lastDataHash = null;
    
    var ACTIVE_DATA_INTERVAL = 2000;
    var HISTORY_DATA_INTERVAL = 10000;
    var CHART_UPDATE_INTERVAL = 5000;
    var WS_RECONNECT_TIMEOUT = 2000;
    
    var currentUserId = null;
    var currentUserFirstName = null;

    function getTelegramUserId() {
        if (TelegramWebApp && TelegramWebApp.initDataUnsafe && TelegramWebApp.initDataUnsafe.user) {
            return TelegramWebApp.initDataUnsafe.user.id;
        }
        var urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('user_id') || null;
    }

    function getTelegramUserFirstName() {
        if (TelegramWebApp && TelegramWebApp.initDataUnsafe && TelegramWebApp.initDataUnsafe.user) {
            return TelegramWebApp.initDataUnsafe.user.first_name || null;
        }
        return null;
    }

    function updateUserGreeting() {
        var greetingEl = document.getElementById('user-greeting');
        if (!greetingEl) return;
        
        if (currentUserFirstName) {
            greetingEl.innerHTML = '👤 Selamat datang, <strong>' + currentUserFirstName + '</strong>';
            greetingEl.className = 'user-greeting authenticated';
        } else if (currentUserId) {
            greetingEl.innerHTML = '👤 User ID: <strong>' + currentUserId + '</strong>';
            greetingEl.className = 'user-greeting authenticated';
        } else {
            greetingEl.innerHTML = '👁️ Mode Tamu';
            greetingEl.className = 'user-greeting guest';
        }
    }

    function initTelegram() {
        debugLog('initTelegram called');
        if (TelegramWebApp) {
            try {
                TelegramWebApp.ready();
                TelegramWebApp.expand();
                debugLog('Telegram WebApp initialized');
                var theme = TelegramWebApp.themeParams;
                if (theme && theme.bg_color) {
                    document.documentElement.style.setProperty('--tg-theme-bg-color', theme.bg_color);
                }
            } catch (e) {
                debugLog('Telegram init error: ' + e.message);
            }
        } else {
            debugLog('Telegram WebApp not available - running standalone');
        }
        
        currentUserId = getTelegramUserId();
        currentUserFirstName = getTelegramUserFirstName();
        debugLog('User ID: ' + (currentUserId || 'anonymous'));
        debugLog('User Name: ' + (currentUserFirstName || 'unknown'));
        updateUserGreeting();
    }

    function setConnectionState(state) {
        connectionState = state;
        var statusDot = document.querySelector('.status-dot');
        var statusText = document.querySelector('.status-text');
        
        if (statusDot && statusText) {
            statusDot.classList.remove('offline', 'connecting', 'live');
            
            switch(state) {
                case 'connected':
                case 'live':
                    statusDot.classList.add('live');
                    statusText.textContent = 'LIVE';
                    statusText.className = 'status-text live';
                    isConnected = true;
                    break;
                case 'connecting':
                case 'reconnecting':
                    statusDot.classList.add('connecting');
                    statusText.textContent = state === 'reconnecting' ? 'MENYAMBUNG ULANG...' : 'MENYAMBUNG...';
                    statusText.className = 'status-text connecting';
                    break;
                case 'offline':
                case 'disconnected':
                default:
                    statusDot.classList.add('offline');
                    statusText.textContent = 'OFFLINE';
                    statusText.className = 'status-text offline';
                    isConnected = false;
                    break;
            }
        }
        debugLog('Connection state: ' + state);
    }

    function fetchWithRetry(url, maxRetries) {
        maxRetries = maxRetries || 3;
        return new Promise(function(resolve, reject) {
            var attempt = 1;
            function doFetch() {
                debugLog('Fetch attempt ' + attempt + '/' + maxRetries + ': ' + url);
                var urlWithCache = url + (url.indexOf('?') > -1 ? '&' : '?') + 't=' + Date.now();
                fetch(urlWithCache, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json',
                        'Cache-Control': 'no-cache'
                    }
                }).then(function(response) {
                    debugLog('Response status: ' + response.status);
                    if (!response.ok) {
                        throw new Error('HTTP ' + response.status);
                    }
                    return response.text();
                }).then(function(text) {
                    debugLog('Response length: ' + text.length + ' chars');
                    if (!text || text.trim() === '') {
                        throw new Error('Empty response body');
                    }
                    try {
                        var data = JSON.parse(text);
                        debugLog('Fetch OK: ' + url + ' - data keys: ' + Object.keys(data).join(', '));
                        resolve(data);
                    } catch (parseError) {
                        debugLog('JSON parse error: ' + parseError.message);
                        throw new Error('JSON parse failed');
                    }
                }).catch(function(error) {
                    debugLog('Attempt ' + attempt + '/' + maxRetries + ' failed: ' + error.message);
                    if (attempt >= maxRetries) {
                        reject(error);
                    } else {
                        attempt++;
                        setTimeout(doFetch, 500 * attempt);
                    }
                });
            }
            doFetch();
        });
    }

    function fetchDashboardData() {
        debugLog('fetchDashboardData called');
        var url = '/api/dashboard';
        if (currentUserId) {
            url += '?user_id=' + encodeURIComponent(currentUserId);
        }
        return fetchWithRetry(url).then(function(data) {
            if (data) {
                debugLog('Dashboard data received: price=' + (data.price ? 'yes' : 'no') + ', user_mode=' + (data.user_mode || 'unknown'));
            }
            return data;
        });
    }

    function fetchCandlesData() {
        debugLog('fetchCandlesData called');
        return fetchWithRetry('/api/candles?timeframe=M1&limit=100').then(function(data) {
            if (data && data.candles) {
                debugLog('Candles data received: ' + data.candles.length + ' candles');
            }
            return data;
        });
    }

    function formatPrice(price) {
        if (price === null || price === undefined || isNaN(price)) return '--';
        return parseFloat(price).toFixed(2);
    }

    function formatPercent(value) {
        if (value === null || value === undefined || isNaN(value)) return '--';
        var num = parseFloat(value);
        var sign = num >= 0 ? '+' : '';
        return sign + num.toFixed(2) + '%';
    }

    function formatCurrency(value) {
        if (value === null || value === undefined || isNaN(value)) return '--';
        var num = parseFloat(value);
        var sign = num >= 0 ? '+' : '';
        return sign + '$' + Math.abs(num).toFixed(2);
    }

    function formatTime(timestamp) {
        if (!timestamp) return '--';
        try {
            var date = new Date(timestamp);
            if (isNaN(date.getTime())) return '--';
            var day = String(date.getDate()).padStart(2, '0');
            var month = String(date.getMonth() + 1).padStart(2, '0');
            var year = date.getFullYear();
            var hours = String(date.getHours()).padStart(2, '0');
            var minutes = String(date.getMinutes()).padStart(2, '0');
            var seconds = String(date.getSeconds()).padStart(2, '0');
            return day + '.' + month + '.' + year + ' ' + hours + ':' + minutes + ':' + seconds;
        } catch (e) {
            return '--';
        }
    }

    function formatTimeShort(date) {
        if (!date) return '--';
        var hours = String(date.getHours()).padStart(2, '0');
        var minutes = String(date.getMinutes()).padStart(2, '0');
        var seconds = String(date.getSeconds()).padStart(2, '0');
        return hours + ':' + minutes + ':' + seconds;
    }

    function updateConnectionStatus(connected) {
        if (connected) {
            setConnectionState('live');
        } else {
            setConnectionState('offline');
        }
    }

    function updatePriceCard(data) {
        debugLog('updatePriceCard called');
        var priceElement = document.getElementById('price-main');
        var changeElement = document.getElementById('price-change');
        var bidElement = document.getElementById('price-bid');
        var askElement = document.getElementById('price-ask');
        var spreadElement = document.getElementById('price-spread');
        var rangeElement = document.getElementById('price-range');

        if (!data || !data.price) {
            debugLog('updatePriceCard: no price data');
            return;
        }

        debugLog('Price data: mid=' + data.price.mid + ', bid=' + data.price.bid);

        if (priceElement) {
            var midPrice = data.price.mid || ((data.price.bid + data.price.ask) / 2);
            var oldPrice = parseFloat(priceElement.textContent) || 0;
            priceElement.textContent = formatPrice(midPrice);
            
            priceElement.classList.remove('price-up', 'price-down');
            if (oldPrice > 0 && midPrice > oldPrice) {
                priceElement.classList.add('price-up');
            } else if (oldPrice > 0 && midPrice < oldPrice) {
                priceElement.classList.add('price-down');
            }
            
            debugLog('Updated price-main: ' + formatPrice(midPrice));
        }
        if (changeElement) {
            var changePercent = data.price.change_percent || 0;
            changeElement.textContent = formatPercent(changePercent);
            changeElement.className = 'price-change ' + (changePercent >= 0 ? 'positive' : 'negative');
        }
        if (bidElement) bidElement.textContent = formatPrice(data.price.bid);
        if (askElement) askElement.textContent = formatPrice(data.price.ask);
        if (spreadElement) {
            var spread = data.price.spread || 0;
            spreadElement.textContent = spread.toFixed(1) + ' pips';
        }
        if (rangeElement && data.price.high && data.price.low) {
            rangeElement.textContent = formatPrice(data.price.low) + ' - ' + formatPrice(data.price.high);
        }
        
        lastPriceData = data.price;
    }

    function updateSignalCard(data) {
        debugLog('updateSignalCard called');
        var signalBadge = document.getElementById('signal-badge');
        var entryElement = document.getElementById('signal-entry');
        var slElement = document.getElementById('signal-sl');
        var tpElement = document.getElementById('signal-tp');
        var timeElement = document.getElementById('signal-time');

        if (!signalBadge) return;

        if (data && data.last_signal && data.last_signal.direction) {
            var direction = data.last_signal.direction.toUpperCase();
            var icon = direction === 'BUY' ? '📈' : '📉';
            signalBadge.innerHTML = icon + ' ' + direction;
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

    function updatePositionCard(data) {
        debugLog('updatePositionCard called');
        var container = document.getElementById('position-container');
        if (!container) return;

        if (data && data.active_position && data.active_position.active) {
            var pos = data.active_position;
            var direction = (pos.direction || 'BUY').toUpperCase();
            var pnl = pos.unrealized_pnl || 0;
            var isProfit = pnl >= 0;
            var pnlPips = pos.current_pnl_pips || 0;
            var distanceToTp = pos.distance_to_tp_pips || 0;
            var distanceToSl = pos.distance_to_sl_pips || 0;
            
            var tpProgress = 0;
            var slProgress = 0;
            var tpAlert = '';
            var slAlert = '';
            
            if (pos.take_profit && pos.entry_price && pos.stop_loss) {
                var totalTpDistance = Math.abs(pos.take_profit - pos.entry_price) * 10;
                var totalSlDistance = Math.abs(pos.entry_price - pos.stop_loss) * 10;
                
                if (totalTpDistance > 0) {
                    tpProgress = Math.min(100, Math.max(0, ((totalTpDistance - Math.abs(distanceToTp)) / totalTpDistance) * 100));
                }
                if (totalSlDistance > 0) {
                    slProgress = Math.min(100, Math.max(0, ((totalSlDistance - Math.abs(distanceToSl)) / totalSlDistance) * 100));
                }
                
                if (tpProgress >= 95) {
                    tpAlert = ' 🎯 HAMPIR TP!';
                }
                if (slProgress >= 95) {
                    slAlert = ' ⚠️ HAMPIR SL!';
                }
            }

            var pnlClass = isProfit ? 'profit' : 'loss';
            var pnlIcon = isProfit ? '📈' : '📉';

            container.innerHTML = '<div class="position-header">' +
                '<span class="position-type ' + direction.toLowerCase() + '">' + (direction === 'BUY' ? '📈' : '📉') + ' ' + direction + '</span>' +
                '<span class="position-pnl ' + pnlClass + '">' + pnlIcon + ' ' + formatCurrency(pnl) + '</span>' +
                '</div>' +
                '<div class="position-details">' +
                '<div class="signal-info-item"><div class="signal-info-label">Entry</div><div class="signal-info-value">' + formatPrice(pos.entry_price) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">SL</div><div class="signal-info-value sl-value">' + formatPrice(pos.sl || pos.stop_loss) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">TP</div><div class="signal-info-value tp-value">' + formatPrice(pos.tp || pos.take_profit) + '</div></div>' +
                '</div>' +
                '<div class="position-details" style="margin-top: 8px;">' +
                '<div class="signal-info-item"><div class="signal-info-label">P/L Pips</div><div class="signal-info-value ' + pnlClass + '">' + (pnlPips >= 0 ? '+' : '') + pnlPips.toFixed(1) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">Ke TP' + tpAlert + '</div><div class="signal-info-value tp-distance">' + (distanceToTp ? distanceToTp.toFixed(1) : '--') + ' pips</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">Ke SL' + slAlert + '</div><div class="signal-info-value sl-distance">' + (distanceToSl ? distanceToSl.toFixed(1) : '--') + ' pips</div></div>' +
                '</div>';
            
            if (tpAlert || slAlert) {
                container.classList.add('position-alert');
            } else {
                container.classList.remove('position-alert');
            }
            
            debugLog('Position updated: ' + direction + ' @ ' + pos.entry_price + ' P/L: ' + formatCurrency(pnl));
        } else {
            container.innerHTML = '<div class="no-position"><p>Tidak ada posisi aktif</p></div>';
            container.classList.remove('position-alert');
            debugLog('No active position');
        }
    }

    function updateStatsCard(data) {
        debugLog('updateStatsCard called');
        var winRateElement = document.getElementById('stat-winrate');
        var totalPnlElement = document.getElementById('stat-pnl');
        var todaySignalsElement = document.getElementById('stat-signals');
        var tradesElement = document.getElementById('stat-trades');

        if (data && data.stats) {
            if (winRateElement) winRateElement.textContent = (data.stats.win_rate || 0).toFixed(1) + '%';
            if (totalPnlElement) {
                var pnl = data.stats.total_pnl || 0;
                totalPnlElement.textContent = formatCurrency(pnl);
                totalPnlElement.className = 'stat-value ' + (pnl >= 0 ? 'positive' : 'negative');
            }
            if (todaySignalsElement) todaySignalsElement.textContent = data.stats.signals_today || 0;
            if (tradesElement) tradesElement.textContent = data.stats.total_trades || 0;
        }
    }

    function fetchTradeHistory() {
        debugLog('fetchTradeHistory called');
        var url = '/api/trade-history?limit=10';
        if (currentUserId) {
            url += '&user_id=' + encodeURIComponent(currentUserId);
        }
        return fetchWithRetry(url).then(function(data) {
            if (data && data.trades) {
                debugLog('Trade history received: ' + data.trades.length + ' trades');
                updateTradeHistoryCard(data.trades);
            }
            return data;
        }).catch(function(error) {
            debugLog('Error fetching trade history: ' + error.message);
        });
    }

    function updateTradeHistoryCard(trades) {
        debugLog('updateTradeHistoryCard called');
        var container = document.getElementById('trade-history-container');
        if (!container) return;

        if (!trades || trades.length === 0) {
            container.innerHTML = '<div class="no-trades">Belum ada riwayat trading</div>';
            return;
        }

        var html = '';
        trades.forEach(function(trade) {
            var signalType = (trade.signal_type || 'BUY').toUpperCase();
            var isProfit = trade.pnl >= 0;
            var pnlText = isProfit ? '+$' + Math.abs(trade.pnl).toFixed(2) : '-$' + Math.abs(trade.pnl).toFixed(2);
            var statusClass = trade.status === 'CLOSED' ? (isProfit ? 'win' : 'loss') : 'open';
            var statusIcon = trade.status === 'CLOSED' ? (isProfit ? '✅' : '❌') : '⏳';
            var statusText = trade.status === 'CLOSED' ? (isProfit ? 'WIN' : 'LOSS') : 'OPEN';
            
            var tradeTime = trade.signal_time ? formatTime(trade.signal_time) : '--';
            
            html += '<div class="trade-history-item">';
            html += '<div class="trade-info">';
            html += '<div class="trade-type ' + signalType.toLowerCase() + '">' + (signalType === 'BUY' ? '📈' : '📉') + ' ' + signalType + '</div>';
            html += '<div class="trade-details">Entry: $' + formatPrice(trade.entry_price) + ' | ' + tradeTime + '</div>';
            html += '</div>';
            html += '<div class="trade-result">';
            if (trade.status === 'CLOSED') {
                html += '<div class="trade-pnl ' + (isProfit ? 'positive' : 'negative') + '">' + pnlText + '</div>';
            }
            html += '<div class="trade-status ' + statusClass + '">' + statusIcon + ' ' + statusText + '</div>';
            html += '</div>';
            html += '</div>';
        });

        container.innerHTML = html;
        debugLog('Trade history updated with ' + trades.length + ' trades');
    }

    function updateRegimeCard(data) {
        debugLog('updateRegimeCard called');
        var container = document.getElementById('regime-tags');
        if (!container) return;

        if (data && data.regime && (data.regime.trend || data.regime.bias)) {
            var tags = [];
            
            if (data.regime.trend) {
                var trendText = String(data.regime.trend).toLowerCase();
                var trendClass = '';
                var trendIcon = '📊';
                
                if (trendText.indexOf('bullish') > -1 || trendText.indexOf('uptrend') > -1 || trendText.indexOf('strong_trend') > -1) {
                    trendClass = 'trend-up';
                    trendIcon = '📈';
                } else if (trendText.indexOf('bearish') > -1 || trendText.indexOf('downtrend') > -1) {
                    trendClass = 'trend-down';
                    trendIcon = '📉';
                } else if (trendText.indexOf('range') > -1 || trendText.indexOf('sideways') > -1 || trendText.indexOf('consolidation') > -1) {
                    trendClass = 'range';
                    trendIcon = '↔️';
                } else if (trendText.indexOf('uncertain') > -1 || trendText.indexOf('weak') > -1) {
                    trendClass = 'uncertain';
                    trendIcon = '❓';
                }
                
                tags.push('<span class="regime-tag ' + trendClass + '">' + trendIcon + ' ' + data.regime.trend + '</span>');
                debugLog('Regime trend: ' + data.regime.trend);
            }
            
            if (data.regime.volatility) {
                var volText = String(data.regime.volatility).toLowerCase();
                var volClass = volText.indexOf('high') > -1 ? 'volatile' : volText.indexOf('low') > -1 ? 'calm' : '';
                var volIcon = volText.indexOf('high') > -1 ? '🔥' : volText.indexOf('low') > -1 ? '😴' : '📊';
                tags.push('<span class="regime-tag ' + volClass + '">' + volIcon + ' ' + data.regime.volatility + '</span>');
                debugLog('Regime volatility: ' + data.regime.volatility);
            }
            
            if (data.regime.bias) {
                var biasClass = data.regime.bias === 'BUY' ? 'trend-up' : data.regime.bias === 'SELL' ? 'trend-down' : 'neutral';
                var biasIcon = data.regime.bias === 'BUY' ? '🟢' : data.regime.bias === 'SELL' ? '🔴' : '⚪';
                tags.push('<span class="regime-tag ' + biasClass + '">' + biasIcon + ' Bias: ' + data.regime.bias + '</span>');
                debugLog('Regime bias: ' + data.regime.bias);
            }
            
            if (data.regime.confidence !== undefined) {
                var confPercent = (parseFloat(data.regime.confidence) * 100).toFixed(0);
                var confClass = confPercent >= 70 ? 'high-conf' : confPercent >= 40 ? 'med-conf' : 'low-conf';
                tags.push('<span class="regime-tag ' + confClass + '">🎯 Kepercayaan: ' + confPercent + '%</span>');
                debugLog('Regime confidence: ' + confPercent + '%');
            }
            
            if (data.regime.session) {
                tags.push('<span class="regime-tag session">🕐 ' + data.regime.session + '</span>');
            }
            
            container.innerHTML = tags.length > 0 ? tags.join('') : '<span class="regime-tag loading">⏳ Menganalisis...</span>';
            debugLog('Regime tags rendered: ' + tags.length);
        } else {
            debugLog('No regime data available');
            container.innerHTML = '<span class="regime-tag loading">⏳ Menganalisis...</span>';
        }
    }

    function updateUpdateTime() {
        var element = document.getElementById('update-time');
        if (element) {
            lastUpdateTime = new Date();
            element.textContent = 'Update terakhir: ' + formatTimeShort(lastUpdateTime) + ' WIB';
        }
    }

    function hideLoading() {
        var overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
            setTimeout(function() { overlay.style.display = 'none'; }, 300);
        }
    }

    function initChart() {
        var container = document.getElementById('chart-container');
        debugLog('initChart: container=' + (container ? 'found' : 'not found') + ', LWC=' + (typeof LightweightCharts !== 'undefined'));

        if (!container || typeof LightweightCharts === 'undefined') {
            return false;
        }

        try {
            chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 300,
                layout: { background: { type: 'solid', color: '#1a1a2e' }, textColor: '#ffffff' },
                grid: { vertLines: { color: '#2a2a4e' }, horzLines: { color: '#2a2a4e' } },
                rightPriceScale: { borderColor: '#2a2a4e' },
                timeScale: { borderColor: '#2a2a4e', timeVisible: true }
            });

            if (typeof chart.addCandlestickSeries === 'function') {
                candleSeries = chart.addCandlestickSeries({
                    upColor: '#22c55e', downColor: '#ef4444',
                    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
                    wickUpColor: '#22c55e', wickDownColor: '#ef4444'
                });
            } else if (typeof chart.addSeries === 'function') {
                candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
                    upColor: '#22c55e', downColor: '#ef4444',
                    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
                    wickUpColor: '#22c55e', wickDownColor: '#ef4444'
                });
            } else {
                debugLog('Chart API not compatible');
                return false;
            }

            window.addEventListener('resize', function() {
                if (chart && container) {
                    chart.applyOptions({ width: container.clientWidth });
                }
            });

            debugLog('Chart initialized successfully');
            return true;
        } catch (e) {
            debugLog('Chart init error: ' + e.message);
            return false;
        }
    }

    function convertToWIB(timestamp) {
        try {
            var date = new Date(timestamp);
            if (isNaN(date.getTime())) return Math.floor(Date.now() / 1000);
            var wibOffset = 7 * 60 * 60 * 1000;
            var utcTime = date.getTime() + (date.getTimezoneOffset() * 60 * 1000);
            return Math.floor((utcTime + wibOffset) / 1000);
        } catch (e) {
            return Math.floor(Date.now() / 1000);
        }
    }

    function clearPriceLines() {
        try {
            if (entryLine && candleSeries) {
                candleSeries.removePriceLine(entryLine);
                entryLine = null;
            }
            if (slLine && candleSeries) {
                candleSeries.removePriceLine(slLine);
                slLine = null;
            }
            if (tpLine && candleSeries) {
                candleSeries.removePriceLine(tpLine);
                tpLine = null;
            }
            if (currentPriceLine && candleSeries) {
                candleSeries.removePriceLine(currentPriceLine);
                currentPriceLine = null;
            }
        } catch (e) {
            debugLog('Error clearing price lines: ' + e.message);
        }
    }
    
    function updatePriceLines(position, currentPrice) {
        if (!candleSeries) return;
        
        clearPriceLines();
        
        try {
            if (currentPrice) {
                currentPriceLine = candleSeries.createPriceLine({
                    price: parseFloat(currentPrice),
                    color: '#888888',
                    lineWidth: 1,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'Sekarang'
                });
            }
            
            if (position && position.entry_price) {
                var isBuy = (position.direction || 'BUY').toUpperCase() === 'BUY';
                
                entryLine = candleSeries.createPriceLine({
                    price: parseFloat(position.entry_price),
                    color: '#3b82f6',
                    lineWidth: 2,
                    lineStyle: 0,
                    axisLabelVisible: true,
                    title: 'Entry'
                });
                debugLog('Entry line created at ' + position.entry_price);
                
                if (position.stop_loss) {
                    slLine = candleSeries.createPriceLine({
                        price: parseFloat(position.stop_loss),
                        color: '#ef4444',
                        lineWidth: 2,
                        lineStyle: 0,
                        axisLabelVisible: true,
                        title: 'SL'
                    });
                    debugLog('SL line created at ' + position.stop_loss);
                }
                
                if (position.take_profit) {
                    tpLine = candleSeries.createPriceLine({
                        price: parseFloat(position.take_profit),
                        color: '#22c55e',
                        lineWidth: 2,
                        lineStyle: 0,
                        axisLabelVisible: true,
                        title: 'TP'
                    });
                    debugLog('TP line created at ' + position.take_profit);
                }
            }
        } catch (e) {
            debugLog('Error updating price lines: ' + e.message);
        }
    }

    function updateCandleChart() {
        debugLog('updateCandleChart called');
        if (!candleSeries || !chart) {
            debugLog('Chart or candleSeries not ready yet');
            return Promise.resolve();
        }
        
        return fetchCandlesData().then(function(data) {
            if (!data || !data.candles || data.candles.length === 0) {
                debugLog('No candles data received');
                return;
            }

            debugLog('Received ' + data.candles.length + ' candles');
            
            var chartData = [];
            var seenTimes = {};
            data.candles.forEach(function(candle) {
                var time = convertToWIB(candle.timestamp);
                if (!seenTimes[time]) {
                    seenTimes[time] = true;
                    try {
                        chartData.push({
                            time: time,
                            open: parseFloat(candle.open),
                            high: parseFloat(candle.high),
                            low: parseFloat(candle.low),
                            close: parseFloat(candle.close)
                        });
                    } catch (e) {
                        debugLog('Error parsing candle: ' + e.message);
                    }
                }
            });

            if (chartData.length === 0) {
                debugLog('No valid candles to render');
                return;
            }
            
            chartData.sort(function(a, b) { return a.time - b.time; });
            
            try {
                candleSeries.setData(chartData);
                chart.timeScale().fitContent();
                debugLog('Chart updated with ' + chartData.length + ' candles');
            } catch (e) {
                debugLog('Error setting chart data: ' + e.message);
            }
            
            updatePriceLines(data.active_position, data.current_price);
        }).catch(function(error) {
            debugLog('Error updating chart: ' + (error ? error.message : 'unknown error'));
        });
    }

    function calculateDataHash(data) {
        if (!data) return null;
        try {
            var hashStr = JSON.stringify({
                price: data.price ? data.price.mid : null,
                signal: data.last_signal ? data.last_signal.direction : null,
                position: data.active_position ? data.active_position.unrealized_pnl : null,
                regime: data.regime ? data.regime.trend : null
            });
            return hashStr;
        } catch (e) {
            return null;
        }
    }

    function refreshData() {
        if (fetchInProgress) {
            debugLog('Refresh skipped - fetch in progress');
            return Promise.resolve();
        }

        fetchInProgress = true;
        debugLog('refreshData() started');

        return fetchDashboardData().then(function(data) {
            if (!data) throw new Error('No data returned from API');

            var newHash = calculateDataHash(data);
            var dataChanged = newHash !== lastDataHash;
            lastDataHash = newHash;

            debugLog('Data received, updating UI... (changed: ' + dataChanged + ')');
            hideLoading();
            setConnectionState('live');
            
            updatePriceCard(data);
            updateSignalCard(data);
            updatePositionCard(data);
            updateStatsCard(data);
            updateRegimeCard(data);
            updateUpdateTime();

            debugLog('UI updates complete');
            
        }).then(function() {
            debugLog('refreshData() completed successfully');
        }).catch(function(error) {
            debugLog('ERROR in refreshData: ' + error.message);
            setConnectionState('offline');
        }).finally(function() {
            fetchInProgress = false;
        });
    }

    window.refreshData = refreshData;

    function getWebSocketUrl() {
        var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return protocol + '//' + window.location.host + '/ws/dashboard';
    }
    
    function connectWebSocket() {
        if (!useWebSocket || websocket) return;
        
        try {
            var wsUrl = getWebSocketUrl();
            debugLog('Connecting to WebSocket: ' + wsUrl);
            setConnectionState('connecting');
            
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function() {
                debugLog('WebSocket connected');
                wsReconnectAttempts = 0;
                setConnectionState('live');
                hideLoading();
            };
            
            websocket.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    
                    updatePriceCard(data);
                    updateSignalCard(data);
                    updatePositionCard(data);
                    updateStatsCard(data);
                    updateRegimeCard(data);
                    updateUpdateTime();
                    
                    setConnectionState('live');
                } catch (e) {
                    debugLog('WebSocket message parse error: ' + e.message);
                }
            };
            
            websocket.onclose = function(event) {
                debugLog('WebSocket closed: ' + event.code);
                websocket = null;
                
                if (wsReconnectAttempts < MAX_WS_RECONNECT_ATTEMPTS) {
                    wsReconnectAttempts++;
                    var delay = Math.min(WS_RECONNECT_TIMEOUT * Math.pow(1.5, wsReconnectAttempts), 30000);
                    debugLog('WebSocket reconnecting in ' + delay + 'ms (attempt ' + wsReconnectAttempts + ')');
                    setConnectionState('reconnecting');
                    wsReconnectTimer = setTimeout(connectWebSocket, delay);
                } else {
                    debugLog('WebSocket max reconnect attempts reached, falling back to polling');
                    useWebSocket = false;
                    startPolling();
                }
            };
            
            websocket.onerror = function(error) {
                debugLog('WebSocket error');
                setConnectionState('offline');
            };
            
        } catch (e) {
            debugLog('WebSocket connection error: ' + e.message);
            useWebSocket = false;
            startPolling();
        }
    }
    
    function disconnectWebSocket() {
        if (wsReconnectTimer) {
            clearTimeout(wsReconnectTimer);
            wsReconnectTimer = null;
        }
        if (websocket) {
            websocket.close();
            websocket = null;
        }
    }
    
    function startPolling() {
        debugLog('Starting polling with auto-refresh');
        
        if (updateInterval) clearInterval(updateInterval);
        if (chartUpdateInterval) clearInterval(chartUpdateInterval);
        if (historyUpdateInterval) clearInterval(historyUpdateInterval);
        
        refreshData();
        updateCandleChart();
        fetchTradeHistory();
        
        updateInterval = setInterval(function() {
            refreshData();
        }, ACTIVE_DATA_INTERVAL);
        debugLog('Active data polling started (interval: ' + ACTIVE_DATA_INTERVAL + 'ms)');
        
        chartUpdateInterval = setInterval(function() {
            updateCandleChart();
        }, CHART_UPDATE_INTERVAL);
        debugLog('Chart update started (interval: ' + CHART_UPDATE_INTERVAL + 'ms)');
        
        historyUpdateInterval = setInterval(function() {
            fetchTradeHistory();
        }, HISTORY_DATA_INTERVAL);
        debugLog('History update started (interval: ' + HISTORY_DATA_INTERVAL + 'ms)');
    }

    function startAutoRefresh() {
        debugLog('startAutoRefresh called');
        
        if (useWebSocket && 'WebSocket' in window) {
            connectWebSocket();
            
            refreshData();
            updateCandleChart();
            fetchTradeHistory();
            
            if (!chartUpdateInterval) {
                chartUpdateInterval = setInterval(updateCandleChart, CHART_UPDATE_INTERVAL);
            }
            if (!historyUpdateInterval) {
                historyUpdateInterval = setInterval(fetchTradeHistory, HISTORY_DATA_INTERVAL);
            }
            
            if (!updateInterval) {
                updateInterval = setInterval(function() {
                    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                        refreshData();
                    }
                }, ACTIVE_DATA_INTERVAL);
            }
        } else {
            startPolling();
        }
    }

    function stopAutoRefresh() {
        debugLog('stopAutoRefresh called');
        disconnectWebSocket();
        
        if (updateInterval) {
            clearInterval(updateInterval);
            updateInterval = null;
        }
        if (chartUpdateInterval) {
            clearInterval(chartUpdateInterval);
            chartUpdateInterval = null;
        }
        if (historyUpdateInterval) {
            clearInterval(historyUpdateInterval);
            historyUpdateInterval = null;
        }
    }

    function init() {
        debugLog('init() called');
        initTelegram();
        initChart();
        startAutoRefresh();
        debugLog('Dashboard initialization complete');

        var refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function() {
                debugLog('Manual refresh triggered');
                refreshData().then(function() {
                    updateCandleChart();
                    fetchTradeHistory();
                });
            });
        }

        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                debugLog('Page hidden - reducing update frequency');
            } else {
                debugLog('Page visible - resuming full updates');
                refreshData();
                updateCandleChart();
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.addEventListener('error', function(event) {
        debugLog('Global error: ' + event.message);
    });
})();
