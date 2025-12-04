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
    
    var FPS_MODE = (function() {
        try {
            var urlParams = new URLSearchParams(window.location.search);
            return urlParams.get('fps') === '1';
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
    debugLog('app.js loaded - v5.1 (Enhanced Dashboard)');

    var isConnected = false;
    var lastUpdateTime = null;
    var updateInterval = null;
    var chartUpdateInterval = null;
    var historyUpdateInterval = null;
    var fetchInProgress = false;
    var chart = null;
    var candleSeries = null;
    var emaSeries9 = null;
    var emaSeries21 = null;
    var emaSeriesInitialized = false;
    var entryLine = null;
    var slLine = null;
    var tpLine = null;
    var currentPriceLine = null;
    var lastActivePosition = null;
    var websocket = null;
    var wsReconnectTimer = null;
    var wsReconnectAttempts = 0;
    var MAX_WS_RECONNECT_ATTEMPTS = 15;
    var useWebSocket = false;
    var connectionState = 'disconnected';
    var lastPriceData = null;
    var lastDataHash = null;
    
    var cachedCandleData = null;
    var lastCandleDataFetch = 0;
    var CANDLE_CACHE_DURATION = 2000;
    
    var ACTIVE_DATA_INTERVAL = 1000;
    var HISTORY_DATA_INTERVAL = 10000;
    var CHART_UPDATE_INTERVAL = 5000;
    var BLOCKING_STATS_INTERVAL = 30000;
    var WS_RECONNECT_TIMEOUT = 2000;
    var SYNC_RETRY_COUNT = 3;
    var SYNC_RETRY_DELAY = 2000;
    var REGIME_TIMEOUT = 5000;
    var MAX_VISIBLE_TRADES = 50;
    var TRADES_PER_PAGE = 20;
    var blockingStatsInterval = null;
    
    var currentUserId = null;
    var currentUserFirstName = null;
    
    var audioAlertEnabled = true;
    var lastAlertTime = 0;
    var ALERT_COOLDOWN = 10000;
    
    var lastSignalId = null;
    var lastPositionStatus = null;
    
    var tpAlertFired = {};
    var slAlertFired = {};
    
    var fpsCounter = {
        frames: 0,
        lastTime: performance.now(),
        fps: 0
    };
    
    var apiResponseTimes = [];
    var MAX_RESPONSE_TIMES = 20;
    
    var tradeHistoryBuffer = [];
    var visibleTradeEnd = 20;
    var isLoadingMoreTrades = false;

    function calculateEMA(data, period) {
        if (!data || data.length < period) return [];
        var k = 2 / (period + 1);
        var emaData = [];
        var sum = 0;
        for (var i = 0; i < period; i++) {
            sum += data[i].close;
        }
        var prevEma = sum / period;
        emaData.push({ time: data[period - 1].time, value: prevEma });
        
        for (var i = period; i < data.length; i++) {
            var ema = (data[i].close - prevEma) * k + prevEma;
            emaData.push({ time: data[i].time, value: ema });
            prevEma = ema;
        }
        return emaData;
    }

    function showToast(message, type, duration) {
        type = type || 'info';
        duration = duration || 3000;
        
        var container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        
        if (container.children.length > 3) {
            return;
        }
        
        var toast = document.createElement('div');
        toast.className = 'toast toast-' + type;
        
        var icon = '';
        switch(type) {
            case 'success': icon = '✅'; break;
            case 'error': icon = '❌'; break;
            case 'warning': icon = '⚠️'; break;
            case 'signal': icon = '📡'; break;
            case 'position': icon = '📊'; break;
            default: icon = 'ℹ️';
        }
        
        toast.innerHTML = '<span class="toast-icon">' + icon + '</span><span class="toast-message">' + message + '</span>';
        container.appendChild(toast);
        
        requestAnimationFrame(function() {
            toast.classList.add('toast-show');
        });
        
        setTimeout(function() {
            toast.classList.remove('toast-show');
            toast.classList.add('toast-hide');
            setTimeout(function() {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    }

    function playAlertSound(type, positionId) {
        if (!audioAlertEnabled) return;
        
        var alertKey = positionId + '_' + type;
        if (type === 'tp' && tpAlertFired[alertKey]) return;
        if (type === 'sl' && slAlertFired[alertKey]) return;
        
        var now = Date.now();
        if (now - lastAlertTime < ALERT_COOLDOWN) return;
        lastAlertTime = now;
        
        if (type === 'tp') tpAlertFired[alertKey] = true;
        if (type === 'sl') slAlertFired[alertKey] = true;
        
        try {
            var audioContext = new (window.AudioContext || window.webkitAudioContext)();
            var oscillator = audioContext.createOscillator();
            var gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            if (type === 'tp') {
                oscillator.frequency.value = 880;
                oscillator.type = 'sine';
            } else if (type === 'sl') {
                oscillator.frequency.value = 440;
                oscillator.type = 'sawtooth';
            } else {
                oscillator.frequency.value = 660;
                oscillator.type = 'square';
            }
            
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.5);
        } catch (e) {
            debugLog('Audio alert error: ' + e.message);
        }
    }

    function resetAlertFlags(positionId) {
        if (positionId) {
            delete tpAlertFired[positionId + '_tp'];
            delete slAlertFired[positionId + '_sl'];
        } else {
            tpAlertFired = {};
            slAlertFired = {};
        }
    }

    function formatPipsAndUSD(pips, usd) {
        var pipsStr = (pips >= 0 ? '+' : '') + pips.toFixed(1) + ' pips';
        var usdStr = (usd >= 0 ? '+' : '') + '$' + Math.abs(usd).toFixed(2);
        return pipsStr + ' / ' + usdStr;
    }

    function calculateTTL(signalTimestamp, maxAge) {
        maxAge = maxAge || 3600000;
        if (!signalTimestamp) return null;
        
        var signalTime = new Date(signalTimestamp).getTime();
        var now = Date.now();
        var elapsed = now - signalTime;
        var remaining = maxAge - elapsed;
        
        if (remaining <= 0) return 'Kadaluarsa';
        
        var minutes = Math.floor(remaining / 60000);
        var seconds = Math.floor((remaining % 60000) / 1000);
        
        if (minutes > 0) {
            return minutes + 'm ' + seconds + 's';
        }
        return seconds + 's';
    }

    function getConfidenceGrade(confidence) {
        if (confidence >= 80) return { grade: 'A', color: 'high-conf', label: 'Sangat Tinggi' };
        if (confidence >= 60) return { grade: 'B', color: 'med-conf', label: 'Tinggi' };
        if (confidence >= 40) return { grade: 'C', color: 'low-conf', label: 'Sedang' };
        return { grade: 'D', color: 'very-low-conf', label: 'Rendah' };
    }

    function updateFPSCounter() {
        if (!FPS_MODE) return;
        
        fpsCounter.frames++;
        var now = performance.now();
        var elapsed = now - fpsCounter.lastTime;
        
        if (elapsed >= 1000) {
            fpsCounter.fps = Math.round((fpsCounter.frames * 1000) / elapsed);
            fpsCounter.frames = 0;
            fpsCounter.lastTime = now;
            
            var fpsEl = document.getElementById('fps-counter');
            if (fpsEl) {
                fpsEl.textContent = 'FPS: ' + fpsCounter.fps;
                fpsEl.className = 'perf-badge ' + (fpsCounter.fps >= 30 ? 'good' : fpsCounter.fps >= 15 ? 'warning' : 'bad');
            }
        }
        
        requestAnimationFrame(updateFPSCounter);
    }

    function updateMemoryUsage() {
        if (!FPS_MODE || !performance.memory) return;
        
        var memEl = document.getElementById('memory-usage');
        if (memEl) {
            var usedMB = Math.round(performance.memory.usedJSHeapSize / 1048576);
            var totalMB = Math.round(performance.memory.totalJSHeapSize / 1048576);
            memEl.textContent = 'MEM: ' + usedMB + '/' + totalMB + ' MB';
            memEl.className = 'perf-badge ' + (usedMB < totalMB * 0.7 ? 'good' : usedMB < totalMB * 0.9 ? 'warning' : 'bad');
        }
    }

    function updateAPIResponseTime(responseTime) {
        apiResponseTimes.push(responseTime);
        if (apiResponseTimes.length > MAX_RESPONSE_TIMES) {
            apiResponseTimes.shift();
        }
        
        if (!FPS_MODE) return;
        
        var avgTime = apiResponseTimes.reduce(function(a, b) { return a + b; }, 0) / apiResponseTimes.length;
        var apiEl = document.getElementById('api-response-time');
        if (apiEl) {
            apiEl.textContent = 'API: ' + Math.round(avgTime) + 'ms';
            apiEl.className = 'perf-badge ' + (avgTime < 200 ? 'good' : avgTime < 500 ? 'warning' : 'bad');
        }
    }

    function getTelegramUserId() {
        if (TelegramWebApp && TelegramWebApp.initDataUnsafe && TelegramWebApp.initDataUnsafe.user) {
            return TelegramWebApp.initDataUnsafe.user.id;
        }
        var urlParams = new URLSearchParams(window.location.search);
        var urlUserId = urlParams.get('user_id');
        if (urlUserId) return urlUserId;
        if (window.currentUserId) return window.currentUserId;
        try {
            var storedUserId = localStorage.getItem('currentUserId');
            if (storedUserId) return storedUserId;
        } catch (e) {}
        return null;
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
            greetingEl.innerHTML = 'Selamat datang, <strong>' + currentUserFirstName + '</strong>';
            greetingEl.className = 'user-greeting authenticated';
        } else if (currentUserId) {
            greetingEl.innerHTML = 'User ID: <strong>' + currentUserId + '</strong>';
            greetingEl.className = 'user-greeting authenticated';
        } else {
            greetingEl.innerHTML = 'Mode Tamu';
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
            statusDot.classList.remove('offline', 'connecting', 'live', 'syncing');
            
            switch(state) {
                case 'connected':
                case 'live':
                    statusDot.classList.add('live');
                    statusText.textContent = 'LIVE';
                    statusText.className = 'status-text live';
                    isConnected = true;
                    break;
                case 'syncing':
                    statusDot.classList.add('syncing');
                    statusText.textContent = 'SINKRONISASI...';
                    statusText.className = 'status-text syncing';
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
        var startTime = performance.now();
        
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
                    var responseTime = performance.now() - startTime;
                    updateAPIResponseTime(responseTime);
                    
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
                        showToast('Koneksi gagal: ' + error.message, 'error');
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

    function fetchCandlesData(forceRefresh) {
        debugLog('fetchCandlesData called');
        
        var now = Date.now();
        if (!forceRefresh && cachedCandleData && (now - lastCandleDataFetch) < CANDLE_CACHE_DURATION) {
            debugLog('Using cached candle data (age: ' + (now - lastCandleDataFetch) + 'ms)');
            return Promise.resolve(cachedCandleData);
        }
        
        var candlesUrl = '/api/candles?timeframe=M1&limit=100';
        if (currentUserId) {
            candlesUrl += '&user_id=' + encodeURIComponent(currentUserId);
        }
        
        return fetchWithRetry(candlesUrl).then(function(data) {
            if (data && data.candles) {
                debugLog('Candles data received: ' + data.candles.length + ' candles');
                cachedCandleData = data;
                lastCandleDataFetch = now;
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
            
            var wibDate = new Date(date.getTime() + (date.getTimezoneOffset() * 60 * 1000) + (7 * 60 * 60 * 1000));
            
            var day = String(wibDate.getDate()).padStart(2, '0');
            var month = String(wibDate.getMonth() + 1).padStart(2, '0');
            var year = wibDate.getFullYear();
            var hours = String(wibDate.getHours()).padStart(2, '0');
            var minutes = String(wibDate.getMinutes()).padStart(2, '0');
            var seconds = String(wibDate.getSeconds()).padStart(2, '0');
            return day + '.' + month + '.' + year + ' ' + hours + ':' + minutes + ':' + seconds + ' WIB';
        } catch (e) {
            return '--';
        }
    }

    function formatTimeShort(date) {
        if (!date) return '--';
        var wibDate = new Date(date.getTime() + (date.getTimezoneOffset() * 60 * 1000) + (7 * 60 * 60 * 1000));
        var hours = String(wibDate.getHours()).padStart(2, '0');
        var minutes = String(wibDate.getMinutes()).padStart(2, '0');
        var seconds = String(wibDate.getSeconds()).padStart(2, '0');
        return hours + ':' + minutes + ':' + seconds + ' WIB';
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
        var ttlElement = document.getElementById('signal-ttl');
        var winRateElement = document.getElementById('signal-winrate');
        var confidenceElement = document.getElementById('signal-confidence');
        var accuracyElement = document.getElementById('signal-accuracy');

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
            
            if (ttlElement) {
                var ttl = calculateTTL(data.last_signal.timestamp, 3600000);
                ttlElement.textContent = ttl || '--';
                ttlElement.className = 'signal-info-value ' + (ttl === 'Kadaluarsa' ? 'expired' : 'active');
            }
            
            if (winRateElement) {
                var winRate = data.signal_stats ? data.signal_stats.win_rate : null;
                winRateElement.textContent = winRate !== null ? winRate.toFixed(1) + '%' : '--';
            }
            
            if (confidenceElement) {
                var confidence = data.last_signal.confidence || (data.regime ? data.regime.confidence * 100 : 50);
                var gradeInfo = getConfidenceGrade(confidence);
                confidenceElement.innerHTML = '<span class="grade-badge ' + gradeInfo.color + '">' + gradeInfo.grade + '</span> ' + gradeInfo.label;
            }
            
            if (accuracyElement) {
                var accuracy = data.signal_stats ? data.signal_stats.entry_accuracy : null;
                accuracyElement.textContent = accuracy !== null ? accuracy.toFixed(1) + '%' : '--';
            }
            
            var signalId = data.last_signal.timestamp + '_' + direction;
            if (lastSignalId !== signalId) {
                showToast('Sinyal baru: ' + direction + ' @ ' + formatPrice(data.last_signal.entry_price), 'signal');
                lastSignalId = signalId;
            }
        } else {
            signalBadge.innerHTML = '⏳ Menunggu Sinyal';
            signalBadge.className = 'signal-badge neutral';
            if (entryElement) entryElement.textContent = '--';
            if (slElement) slElement.textContent = '--';
            if (tpElement) tpElement.textContent = '--';
            if (timeElement) timeElement.textContent = '--';
            if (ttlElement) ttlElement.textContent = '--';
            if (winRateElement) winRateElement.textContent = '--';
            if (confidenceElement) confidenceElement.textContent = '--';
            if (accuracyElement) accuracyElement.textContent = '--';
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
            
            var positionId = pos.id || pos.entry_price + '_' + pos.signal_time;
            
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
                    tpAlert = ' 🎯';
                    if (audioAlertEnabled) {
                        playAlertSound('tp', positionId);
                    }
                }
                if (slProgress >= 95) {
                    slAlert = ' ⚠️';
                    if (audioAlertEnabled) {
                        playAlertSound('sl', positionId);
                    }
                }
            }

            var pnlClass = isProfit ? 'profit' : 'loss';
            var pnlIcon = isProfit ? '📈' : '📉';
            
            var trailingSL = pos.trailing_sl || pos.stop_loss;
            var hasTrailing = pos.trailing_sl && pos.trailing_sl !== pos.stop_loss;

            container.innerHTML = '<div class="position-header">' +
                '<span class="position-type ' + direction.toLowerCase() + '">' + (direction === 'BUY' ? '📈' : '📉') + ' ' + direction + '</span>' +
                '<span class="position-pnl ' + pnlClass + '">' + pnlIcon + ' ' + formatPipsAndUSD(pnlPips, pnl) + '</span>' +
                '</div>' +
                '<div class="position-details">' +
                '<div class="signal-info-item"><div class="signal-info-label">Entry</div><div class="signal-info-value">' + formatPrice(pos.entry_price) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">SL' + (hasTrailing ? ' 🔄' : '') + '</div><div class="signal-info-value sl-value">' + formatPrice(trailingSL) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">TP</div><div class="signal-info-value tp-value">' + formatPrice(pos.tp || pos.take_profit) + '</div></div>' +
                '</div>' +
                '<div class="progress-section">' +
                '<div class="progress-item">' +
                '<div class="progress-label">Ke TP' + tpAlert + '</div>' +
                '<div class="progress-bar-container">' +
                '<div class="progress-bar progress-tp" style="width: ' + tpProgress + '%"></div>' +
                '</div>' +
                '<div class="progress-value">' + tpProgress.toFixed(0) + '% (' + (distanceToTp ? distanceToTp.toFixed(1) : '--') + ' pips)</div>' +
                '</div>' +
                '<div class="progress-item">' +
                '<div class="progress-label">Ke SL' + slAlert + '</div>' +
                '<div class="progress-bar-container">' +
                '<div class="progress-bar progress-sl" style="width: ' + slProgress + '%"></div>' +
                '</div>' +
                '<div class="progress-value">' + slProgress.toFixed(0) + '% (' + (distanceToSl ? distanceToSl.toFixed(1) : '--') + ' pips)</div>' +
                '</div>' +
                '</div>' +
                '<div class="audio-toggle">' +
                '<label class="toggle-label">' +
                '<input type="checkbox" id="audio-alert-toggle" ' + (audioAlertEnabled ? 'checked' : '') + '>' +
                '<span class="toggle-text">🔔 Audio Alert 95%</span>' +
                '</label>' +
                '</div>';
            
            var audioToggle = document.getElementById('audio-alert-toggle');
            if (audioToggle) {
                audioToggle.addEventListener('change', function() {
                    audioAlertEnabled = this.checked;
                    showToast('Audio alert ' + (audioAlertEnabled ? 'aktif' : 'nonaktif'), 'info');
                });
            }
            
            if (tpAlert || slAlert) {
                container.classList.add('position-alert');
            } else {
                container.classList.remove('position-alert');
            }
            
            if (lastPositionStatus === 'closed') {
                showToast('Posisi baru dibuka: ' + direction + ' @ ' + formatPrice(pos.entry_price), 'position');
                resetAlertFlags();
            }
            lastPositionStatus = 'open';
            
            debugLog('Position updated: ' + direction + ' @ ' + pos.entry_price + ' P/L: ' + formatCurrency(pnl));
        } else {
            container.innerHTML = '<div class="no-position"><p>Tidak ada posisi aktif</p></div>';
            container.classList.remove('position-alert');
            
            if (lastPositionStatus === 'open') {
                showToast('Posisi telah ditutup', 'position');
                resetAlertFlags();
            }
            lastPositionStatus = 'closed';
            
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
        var url = '/api/trade-history?limit=' + MAX_VISIBLE_TRADES;
        if (currentUserId) {
            url += '&user_id=' + encodeURIComponent(currentUserId);
        }
        return fetchWithRetry(url).then(function(data) {
            if (data && data.trades) {
                debugLog('Trade history received: ' + data.trades.length + ' trades');
                tradeHistoryBuffer = data.trades;
                renderVisibleTrades();
            }
            return data;
        }).catch(function(error) {
            debugLog('Error fetching trade history: ' + error.message);
        });
    }

    function renderVisibleTrades() {
        debugLog('renderVisibleTrades called');
        var container = document.getElementById('trade-history-container');
        if (!container) return;

        if (!tradeHistoryBuffer || tradeHistoryBuffer.length === 0) {
            container.innerHTML = '<div class="no-trades">Belum ada riwayat trading</div>';
            return;
        }

        var endIndex = Math.min(visibleTradeEnd, tradeHistoryBuffer.length);
        var visibleTrades = tradeHistoryBuffer.slice(0, endIndex);

        var html = '<div class="trade-list" id="trade-list">';
        visibleTrades.forEach(function(trade, index) {
            var signalType = (trade.signal_type || 'BUY').toUpperCase();
            var isProfit = trade.pnl >= 0;
            var pnlText = isProfit ? '+$' + Math.abs(trade.pnl).toFixed(2) : '-$' + Math.abs(trade.pnl).toFixed(2);
            var statusClass = trade.status === 'CLOSED' ? (isProfit ? 'win' : 'loss') : 'open';
            var statusIcon = trade.status === 'CLOSED' ? (isProfit ? '✅' : '❌') : '⏳';
            var statusText = trade.status === 'CLOSED' ? (isProfit ? 'WIN' : 'LOSS') : 'OPEN';
            
            var tradeTime = trade.signal_time ? formatTime(trade.signal_time) : '--';
            
            html += '<div class="trade-history-item" data-index="' + index + '">';
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
        html += '</div>';
        
        var remaining = tradeHistoryBuffer.length - endIndex;
        if (remaining > 0) {
            var loadCount = Math.min(TRADES_PER_PAGE, remaining);
            html += '<div class="load-more-container">';
            html += '<button class="load-more-btn" id="load-more-trades">Muat ' + loadCount + ' trade lagi (' + remaining + ' tersisa)</button>';
            html += '</div>';
        }

        container.innerHTML = html;
        
        var loadMoreBtn = document.getElementById('load-more-trades');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', function() {
                if (!isLoadingMoreTrades) {
                    isLoadingMoreTrades = true;
                    visibleTradeEnd = Math.min(visibleTradeEnd + TRADES_PER_PAGE, tradeHistoryBuffer.length);
                    renderVisibleTrades();
                    isLoadingMoreTrades = false;
                    debugLog('Loaded more trades, now showing ' + visibleTradeEnd);
                }
            });
        }
        
        debugLog('Trade history rendered: ' + visibleTrades.length + ' of ' + tradeHistoryBuffer.length + ' trades');
    }

    function updateTradeHistoryCard(trades) {
        tradeHistoryBuffer = trades;
        visibleTradeEnd = Math.min(20, trades.length);
        renderVisibleTrades();
    }

    var regimeTimeout = null;
    var lastRegimeUpdate = 0;

    function updateRegimeCard(data) {
        debugLog('updateRegimeCard called');
        var container = document.getElementById('regime-tags');
        if (!container) return;

        if (regimeTimeout) {
            clearTimeout(regimeTimeout);
        }

        if (data && data.regime && (data.regime.trend || data.regime.bias)) {
            lastRegimeUpdate = Date.now();
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
                
                tags.push('<span class="regime-tag ' + trendClass + ' regime-animate">' + trendIcon + ' ' + data.regime.trend + '</span>');
                debugLog('Regime trend: ' + data.regime.trend);
            }
            
            if (data.regime.volatility) {
                var volText = String(data.regime.volatility).toLowerCase();
                var volClass = volText.indexOf('high') > -1 ? 'volatile' : volText.indexOf('low') > -1 ? 'calm' : '';
                var volIcon = volText.indexOf('high') > -1 ? '🔥' : volText.indexOf('low') > -1 ? '😴' : '📊';
                tags.push('<span class="regime-tag ' + volClass + ' regime-animate">' + volIcon + ' ' + data.regime.volatility + '</span>');
                debugLog('Regime volatility: ' + data.regime.volatility);
            }
            
            if (data.regime.bias) {
                var biasClass = data.regime.bias === 'BUY' ? 'trend-up' : data.regime.bias === 'SELL' ? 'trend-down' : 'neutral';
                var biasIcon = data.regime.bias === 'BUY' ? '🟢' : data.regime.bias === 'SELL' ? '🔴' : '⚪';
                tags.push('<span class="regime-tag ' + biasClass + ' regime-animate">' + biasIcon + ' Bias: ' + data.regime.bias + '</span>');
                debugLog('Regime bias: ' + data.regime.bias);
            }
            
            if (data.regime.confidence !== undefined) {
                var confPercent = (parseFloat(data.regime.confidence) * 100).toFixed(0);
                var confClass = confPercent >= 80 ? 'high-conf' : confPercent >= 50 ? 'med-conf' : 'low-conf';
                tags.push('<span class="regime-tag ' + confClass + ' regime-animate">🎯 Kepercayaan: ' + confPercent + '%</span>');
                debugLog('Regime confidence: ' + confPercent + '%');
            }
            
            if (data.regime.session) {
                tags.push('<span class="regime-tag session regime-animate">🕐 ' + data.regime.session + '</span>');
            }
            
            container.innerHTML = tags.length > 0 ? tags.join('') : '<span class="regime-tag loading">⏳ Menganalisis...</span>';
            debugLog('Regime tags rendered: ' + tags.length);
        } else {
            var timeSinceLastUpdate = Date.now() - lastRegimeUpdate;
            if (timeSinceLastUpdate > REGIME_TIMEOUT || lastRegimeUpdate === 0) {
                debugLog('No regime data available - showing timeout message');
                container.innerHTML = '<span class="regime-tag uncertain regime-animate">❓ Data tidak tersedia</span>';
            } else {
                debugLog('Waiting for regime data...');
                container.innerHTML = '<span class="regime-tag loading">⏳ Menganalisis...</span>';
            }
        }
        
        regimeTimeout = setTimeout(function() {
            var timeSinceLastUpdate = Date.now() - lastRegimeUpdate;
            if (timeSinceLastUpdate > REGIME_TIMEOUT) {
                container.innerHTML = '<span class="regime-tag uncertain regime-animate">❓ Timeout - coba refresh</span>';
            }
        }, REGIME_TIMEOUT);
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
        if (!container) {
            debugLog('Chart container not found');
            return false;
        }

        debugLog('Initializing chart...');
        try {
            if (typeof LightweightCharts === 'undefined') {
                debugLog('LightweightCharts not loaded');
                container.innerHTML = '<div class="chart-error">Chart library tidak tersedia</div>';
                return false;
            }

            var chartWidth = container.clientWidth || 350;
            var chartHeight = 300;

            if (LightweightCharts.createChart) {
                chart = LightweightCharts.createChart(container, {
                    width: chartWidth,
                    height: chartHeight,
                    layout: {
                        background: { type: 'solid', color: '#1a1a2e' },
                        textColor: '#d1d4dc',
                    },
                    grid: {
                        vertLines: { color: 'rgba(42, 46, 57, 0.6)' },
                        horzLines: { color: 'rgba(42, 46, 57, 0.6)' },
                    },
                    crosshair: {
                        mode: LightweightCharts.CrosshairMode.Normal,
                    },
                    rightPriceScale: {
                        borderColor: 'rgba(197, 203, 206, 0.4)',
                        scaleMargins: {
                            top: 0.1,
                            bottom: 0.1,
                        },
                    },
                    timeScale: {
                        borderColor: 'rgba(197, 203, 206, 0.4)',
                        timeVisible: true,
                        secondsVisible: false,
                    },
                    handleScale: {
                        axisPressedMouseMove: {
                            time: true,
                            price: true,
                        },
                    },
                    handleScroll: {
                        vertTouchDrag: true,
                        horzTouchDrag: true,
                        mouseWheel: true,
                        pressedMouseMove: true,
                    },
                });

                candleSeries = chart.addCandlestickSeries({
                    upColor: '#22c55e',
                    downColor: '#ef4444',
                    borderDownColor: '#ef4444',
                    borderUpColor: '#22c55e',
                    wickDownColor: '#ef4444',
                    wickUpColor: '#22c55e',
                });

                emaSeries9 = chart.addLineSeries({
                    color: '#3b82f6',
                    lineWidth: 1,
                    title: 'EMA 9',
                    crosshairMarkerVisible: false,
                });

                emaSeries21 = chart.addLineSeries({
                    color: '#fbbf24',
                    lineWidth: 1,
                    title: 'EMA 21',
                    crosshairMarkerVisible: false,
                });

                emaSeriesInitialized = true;
                debugLog('Chart created with createChart API');
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
                    color: '#ffd700',
                    lineWidth: 1,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'Now'
                });
            }
            
            if (position && position.active) {
                entryLine = candleSeries.createPriceLine({
                    price: parseFloat(position.entry_price),
                    color: '#3b82f6',
                    lineWidth: 2,
                    lineStyle: 0,
                    axisLabelVisible: true,
                    title: 'Entry'
                });
                debugLog('Entry line created at ' + position.entry_price);
                
                var slPrice = position.trailing_sl || position.stop_loss;
                if (slPrice) {
                    slLine = candleSeries.createPriceLine({
                        price: parseFloat(slPrice),
                        color: '#ef4444',
                        lineWidth: 2,
                        lineStyle: position.trailing_sl ? 2 : 0,
                        axisLabelVisible: true,
                        title: position.trailing_sl ? 'TSL' : 'SL'
                    });
                    debugLog('SL line created at ' + slPrice);
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

    function updateCandleChart(forceRefresh) {
        debugLog('updateCandleChart called');
        if (!candleSeries || !emaSeriesInitialized) {
            debugLog('Chart series not ready');
            return;
        }

        fetchCandlesData(forceRefresh).then(function(data) {
            if (!data || !data.candles || data.candles.length === 0) {
                debugLog('No candle data received');
                return;
            }

            var formattedCandles = data.candles.map(function(c) {
                return {
                    time: convertToWIB(c.time || c.timestamp),
                    open: parseFloat(c.open),
                    high: parseFloat(c.high),
                    low: parseFloat(c.low),
                    close: parseFloat(c.close)
                };
            }).filter(function(c) {
                return !isNaN(c.time) && !isNaN(c.open) && !isNaN(c.high) && !isNaN(c.low) && !isNaN(c.close);
            }).sort(function(a, b) {
                return a.time - b.time;
            });

            if (formattedCandles.length > 0) {
                candleSeries.setData(formattedCandles);
                
                var ema9Data = calculateEMA(formattedCandles, 9);
                var ema21Data = calculateEMA(formattedCandles, 21);
                
                if (ema9Data.length > 0 && emaSeries9) {
                    emaSeries9.setData(ema9Data);
                }
                if (ema21Data.length > 0 && emaSeries21) {
                    emaSeries21.setData(ema21Data);
                }
                
                var currentPrice = formattedCandles[formattedCandles.length - 1].close;
                updatePriceLines(lastActivePosition, currentPrice);
                
                chart.timeScale().fitContent();
                debugLog('Chart updated with ' + formattedCandles.length + ' candles + EMA overlay');
            }
        }).catch(function(error) {
            debugLog('Chart update error: ' + error.message);
        });
    }

    function refreshData() {
        if (fetchInProgress) {
            debugLog('Fetch already in progress, skipping');
            return Promise.resolve();
        }
        fetchInProgress = true;
        debugLog('refreshData called');
        
        if (!isConnected) {
            setConnectionState('syncing');
        }

        return fetchDashboardData().then(function(data) {
            if (data) {
                setConnectionState('live');
                updatePriceCard(data);
                updateSignalCard(data);
                updatePositionCard(data);
                updateStatsCard(data);
                updateRegimeCard(data);
                updateUpdateTime();
                
                if (data.active_position) {
                    lastActivePosition = data.active_position;
                } else {
                    lastActivePosition = null;
                }
                
                hideLoading();
            }
            fetchInProgress = false;
            return data;
        }).catch(function(error) {
            debugLog('Dashboard refresh error: ' + error.message);
            setConnectionState('offline');
            fetchInProgress = false;
        });
    }

    function closeWebSocket() {
        if (wsReconnectTimer) {
            clearTimeout(wsReconnectTimer);
            wsReconnectTimer = null;
        }
        if (websocket) {
            websocket.close();
            websocket = null;
        }
    }
    
    function fetchBlockingStats() {
        var url = '/api/blocking-stats';
        if (currentUserId) {
            url += '?user_id=' + currentUserId;
        }
        
        fetch(url)
            .then(function(response) {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(function(data) {
                debugLog('Blocking stats received: ' + JSON.stringify(data));
                updateBlockingStatsUI(data);
            })
            .catch(function(error) {
                debugLog('Error fetching blocking stats: ' + error.message);
            });
    }
    
    function updateBlockingStatsUI(data) {
        var totalEl = document.getElementById('blocking-total');
        var rateValueEl = document.getElementById('blocking-rate-value');
        var byQualityEl = document.getElementById('blocking-by-quality');
        var byWinRateEl = document.getElementById('blocking-by-winrate');
        var progressEl = document.getElementById('blocking-progress');
        var ratePercentEl = document.getElementById('blocking-rate-percent');
        var alertEl = document.getElementById('blocking-alert');
        
        var gradeAEl = document.getElementById('grade-a-count');
        var gradeBEl = document.getElementById('grade-b-count');
        var gradeCEl = document.getElementById('grade-c-count');
        var gradeDEl = document.getElementById('grade-d-count');
        var gradeFEl = document.getElementById('grade-f-count');
        
        if (totalEl) {
            animateValue(totalEl, parseInt(totalEl.textContent) || 0, data.total_blocked || 0);
        }
        if (rateValueEl) {
            var rateVal = (data.blocking_rate || 0).toFixed(1) + '%';
            rateValueEl.textContent = rateVal;
            rateValueEl.classList.remove('high-rate', 'medium-rate', 'low-rate');
            if (data.blocking_rate > 70) {
                rateValueEl.classList.add('high-rate');
            } else if (data.blocking_rate > 40) {
                rateValueEl.classList.add('medium-rate');
            } else {
                rateValueEl.classList.add('low-rate');
            }
        }
        if (byQualityEl) {
            animateValue(byQualityEl, parseInt(byQualityEl.textContent) || 0, data.blocked_by_quality || 0);
        }
        if (byWinRateEl) {
            animateValue(byWinRateEl, parseInt(byWinRateEl.textContent) || 0, data.blocked_by_win_rate || 0);
        }
        
        if (progressEl) {
            var rate = Math.min(data.blocking_rate || 0, 100);
            progressEl.style.width = rate + '%';
            progressEl.classList.remove('high', 'medium', 'low');
            if (rate > 70) {
                progressEl.classList.add('high');
            } else if (rate > 40) {
                progressEl.classList.add('medium');
            } else {
                progressEl.classList.add('low');
            }
        }
        if (ratePercentEl) {
            ratePercentEl.textContent = (data.blocking_rate || 0).toFixed(1) + '%';
        }
        
        if (alertEl) {
            if (data.blocking_rate > 70) {
                alertEl.classList.remove('hidden');
            } else {
                alertEl.classList.add('hidden');
            }
        }
        
        var breakdown = data.quality_breakdown || {};
        if (gradeAEl) gradeAEl.textContent = breakdown.A || 0;
        if (gradeBEl) gradeBEl.textContent = breakdown.B || 0;
        if (gradeCEl) gradeCEl.textContent = breakdown.C || 0;
        if (gradeDEl) gradeDEl.textContent = breakdown.D || 0;
        if (gradeFEl) gradeFEl.textContent = breakdown.F || 0;
    }
    
    function animateValue(element, start, end) {
        if (start === end) return;
        var duration = 500;
        var startTime = null;
        
        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            var progress = Math.min((timestamp - startTime) / duration, 1);
            var current = Math.floor(progress * (end - start) + start);
            element.textContent = current;
            element.classList.add('value-changed');
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                element.textContent = end;
                setTimeout(function() {
                    element.classList.remove('value-changed');
                }, 300);
            }
        }
        
        window.requestAnimationFrame(step);
    }
    
    function startPolling() {
        debugLog('Starting polling with auto-refresh');
        
        if (updateInterval) clearInterval(updateInterval);
        if (chartUpdateInterval) clearInterval(chartUpdateInterval);
        if (historyUpdateInterval) clearInterval(historyUpdateInterval);
        if (blockingStatsInterval) clearInterval(blockingStatsInterval);
        
        refreshData();
        updateCandleChart(true);
        fetchTradeHistory();
        fetchBlockingStats();
        
        updateInterval = setInterval(function() {
            refreshData();
        }, ACTIVE_DATA_INTERVAL);
        
        chartUpdateInterval = setInterval(function() {
            updateCandleChart(false);
        }, CHART_UPDATE_INTERVAL);
        
        historyUpdateInterval = setInterval(function() {
            fetchTradeHistory();
        }, HISTORY_DATA_INTERVAL);
        
        blockingStatsInterval = setInterval(function() {
            fetchBlockingStats();
        }, BLOCKING_STATS_INTERVAL);
        
        debugLog('Polling started - data: ' + ACTIVE_DATA_INTERVAL + 'ms, chart: ' + CHART_UPDATE_INTERVAL + 'ms, history: ' + HISTORY_DATA_INTERVAL + 'ms, blocking: ' + BLOCKING_STATS_INTERVAL + 'ms');
    }

    function startAutoRefresh() {
        debugLog('startAutoRefresh called');
        startPolling();
    }

    function stopAutoRefresh() {
        debugLog('stopAutoRefresh called');
        closeWebSocket();
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
        if (blockingStatsInterval) {
            clearInterval(blockingStatsInterval);
            blockingStatsInterval = null;
        }
    }

    function initPerformanceMonitor() {
        if (!FPS_MODE) return;
        
        var perfContainer = document.getElementById('perf-monitor');
        if (!perfContainer) {
            perfContainer = document.createElement('div');
            perfContainer.id = 'perf-monitor';
            perfContainer.className = 'perf-monitor';
            perfContainer.innerHTML = 
                '<span class="perf-badge" id="fps-counter">FPS: --</span>' +
                '<span class="perf-badge" id="memory-usage">MEM: --</span>' +
                '<span class="perf-badge" id="api-response-time">API: --</span>';
            document.body.appendChild(perfContainer);
        }
        
        requestAnimationFrame(updateFPSCounter);
        setInterval(updateMemoryUsage, 2000);
        
        debugLog('Performance monitor initialized');
    }

    function init() {
        debugLog('init() called');
        initTelegram();
        initChart();
        initPerformanceMonitor();
        startAutoRefresh();
        debugLog('Dashboard initialization complete');

        var refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function() {
                debugLog('Manual refresh triggered');
                showToast('Memuat ulang data...', 'info');
                cachedCandleData = null;
                lastCandleDataFetch = 0;
                refreshData().then(function() {
                    updateCandleChart(true);
                    fetchTradeHistory();
                    showToast('Data berhasil diperbarui', 'success');
                });
            });
        }

        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                debugLog('Page hidden - reducing update frequency');
            } else {
                debugLog('Page visible - resuming full updates');
                cachedCandleData = null;
                lastCandleDataFetch = 0;
                refreshData();
                updateCandleChart(true);
                fetchTradeHistory();
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
