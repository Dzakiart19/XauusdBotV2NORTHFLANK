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
    debugLog('app.js loaded - v3.0 (IIFE)');

    var isConnected = false;
    var lastUpdateTime = null;
    var updateInterval = null;
    var fetchInProgress = false;
    var chart = null;
    var candleSeries = null;
    var entryLine = null;
    var slLine = null;
    var tpLine = null;
    var currentPriceLine = null;
    var lastActivePosition = null;

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
                        setTimeout(doFetch, 1000 * attempt);
                    }
                });
            }
            doFetch();
        });
    }

    function fetchDashboardData() {
        debugLog('fetchDashboardData called');
        return fetchWithRetry('/api/dashboard').then(function(data) {
            if (data) {
                debugLog('Dashboard data received: price=' + (data.price ? 'yes' : 'no'));
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
            return date.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        } catch (e) {
            return '--';
        }
    }

    function updateConnectionStatus(connected) {
        isConnected = connected;
        var statusDot = document.querySelector('.status-dot');
        var statusText = document.querySelector('.status-text');
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
            priceElement.textContent = formatPrice(midPrice);
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

            container.innerHTML = '<div class="position-header">' +
                '<span class="position-type ' + direction.toLowerCase() + '">' + (direction === 'BUY' ? '📈' : '📉') + ' ' + direction + '</span>' +
                '<span class="position-pnl ' + (isProfit ? 'positive' : 'negative') + '">' + formatCurrency(pnl) + '</span>' +
                '</div>' +
                '<div class="position-details">' +
                '<div class="signal-info-item"><div class="signal-info-label">Entry</div><div class="signal-info-value">' + formatPrice(pos.entry_price) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">SL</div><div class="signal-info-value" style="color: #ef4444;">' + formatPrice(pos.sl || pos.stop_loss) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">TP</div><div class="signal-info-value" style="color: #ffd700;">' + formatPrice(pos.tp || pos.take_profit) + '</div></div>' +
                '</div>' +
                '<div class="position-details" style="margin-top: 8px;">' +
                '<div class="signal-info-item"><div class="signal-info-label">P/L Pips</div><div class="signal-info-value ' + (pnlPips >= 0 ? 'positive' : 'negative') + '">' + (pnlPips >= 0 ? '+' : '') + pnlPips.toFixed(1) + '</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">To TP</div><div class="signal-info-value" style="color: #ffd700;">' + (pos.distance_to_tp_pips ? pos.distance_to_tp_pips.toFixed(1) : '--') + ' pips</div></div>' +
                '<div class="signal-info-item"><div class="signal-info-label">To SL</div><div class="signal-info-value" style="color: #ef4444;">' + (pos.distance_to_sl_pips ? pos.distance_to_sl_pips.toFixed(1) : '--') + ' pips</div></div>' +
                '</div>';
            debugLog('Position updated: ' + direction + ' @ ' + pos.entry_price);
        } else {
            container.innerHTML = '<div class="no-position"><p>Tidak ada posisi aktif</p></div>';
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

    function updateRegimeCard(data) {
        debugLog('updateRegimeCard called');
        var container = document.getElementById('regime-tags');
        if (!container) return;

        if (data && data.regime) {
            var tags = [];
            if (data.regime.trend) {
                var trendClass = data.regime.trend.toLowerCase().indexOf('bullish') > -1 ? 'trend-up' :
                                data.regime.trend.toLowerCase().indexOf('bearish') > -1 ? 'trend-down' : '';
                tags.push('<span class="regime-tag ' + trendClass + '">' + data.regime.trend + '</span>');
            }
            if (data.regime.volatility) {
                var volClass = data.regime.volatility.toLowerCase().indexOf('high') > -1 ? 'volatile' : '';
                tags.push('<span class="regime-tag ' + volClass + '">' + data.regime.volatility + '</span>');
            }
            if (data.regime.bias) {
                var biasClass = data.regime.bias === 'BUY' ? 'trend-up' : data.regime.bias === 'SELL' ? 'trend-down' : '';
                tags.push('<span class="regime-tag ' + biasClass + '">Bias: ' + data.regime.bias + '</span>');
            }
            if (data.regime.confidence !== undefined) {
                var confPercent = (data.regime.confidence * 100).toFixed(0);
                tags.push('<span class="regime-tag">Confidence: ' + confPercent + '%</span>');
            }
            container.innerHTML = tags.length > 0 ? tags.join('') : '<span class="regime-tag">Data tidak tersedia</span>';
        } else {
            container.innerHTML = '<span class="regime-tag">Data tidak tersedia</span>';
        }
    }

    function updateUpdateTime() {
        var element = document.getElementById('update-time');
        if (element) {
            lastUpdateTime = new Date();
            element.textContent = 'Update terakhir: ' + formatTime(lastUpdateTime);
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
                    title: 'Current'
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
        return fetchCandlesData().then(function(data) {
            if (!data || !candleSeries || !data.candles || data.candles.length === 0) {
                debugLog('No candles data or candleSeries not ready');
                return;
            }

            var chartData = [];
            var seenTimes = {};
            data.candles.forEach(function(candle) {
                var time = convertToWIB(candle.timestamp);
                if (!seenTimes[time]) {
                    seenTimes[time] = true;
                    chartData.push({
                        time: time,
                        open: parseFloat(candle.open),
                        high: parseFloat(candle.high),
                        low: parseFloat(candle.low),
                        close: parseFloat(candle.close)
                    });
                }
            });

            chartData.sort(function(a, b) { return a.time - b.time; });
            candleSeries.setData(chartData);
            debugLog('Chart updated with ' + chartData.length + ' candles');
            
            updatePriceLines(data.active_position, data.current_price);
        }).catch(function(error) {
            debugLog('Error updating chart: ' + error.message);
        });
    }

    function refreshData() {
        if (fetchInProgress) {
            debugLog('Refresh skipped - fetch in progress');
            return Promise.resolve();
        }

        fetchInProgress = true;
        debugLog('refreshData() started');

        var refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) refreshBtn.disabled = true;

        return fetchDashboardData().then(function(data) {
            if (!data) throw new Error('No data returned from API');

            debugLog('Data received, updating UI...');
            hideLoading();
            updateConnectionStatus(true);
            updatePriceCard(data);
            updateSignalCard(data);
            updatePositionCard(data);
            updateStatsCard(data);
            updateRegimeCard(data);
            updateUpdateTime();

            debugLog('UI updates complete');
            return updateCandleChart();
        }).then(function() {
            debugLog('refreshData() completed successfully');
        }).catch(function(error) {
            debugLog('ERROR in refreshData: ' + error.message);
            updateConnectionStatus(false);
        }).finally(function() {
            fetchInProgress = false;
            if (refreshBtn) refreshBtn.disabled = false;
        });
    }

    window.refreshData = refreshData;

    function startAutoRefresh() {
        debugLog('startAutoRefresh called');
        if (updateInterval) clearInterval(updateInterval);
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
                refreshData();
            });
        }

        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                debugLog('Page hidden - stopping auto-refresh');
                stopAutoRefresh();
            } else {
                debugLog('Page visible - starting auto-refresh');
                startAutoRefresh();
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
