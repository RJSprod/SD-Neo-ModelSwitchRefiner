// Forge Assistant -- the page's copy of the conversation, and how it stays true.
//
// One store per page. It holds what the server has said, what this page has
// typed, and which requests are in flight; every view reads from it and no view
// writes to it. That separation is not tidiness -- it is what makes "an event
// for thread A cannot touch thread B's draft" a property of one file rather
// than a rule four files have to remember.
//
// Three things here are load-bearing and none of them is obvious.
//
// *Cursors, not callbacks.* A browser's connection always eventually drops: a
// laptop lid, a phone's radio, a proxy's idle timeout. With a callback the
// answer to "what did I miss" is "nobody knows". With a monotonic cursor it is
// "resume after 4,117", and the only case left -- 4,117 has been trimmed out of
// the server's ring -- is detectable and answered with a fresh snapshot rather
// than with a guess.
//
// *A latched operation id.* Send mints one id, once, before anything is sent.
// A duplicate click reuses it; a retry after a lost response reuses it; the
// server recognises it and returns the first attempt's outcome. The page never
// mints a second id for the same intention, because that is exactly how one
// message becomes two.
//
// *Drafts are keyed by conversation and never silently evicted.* What somebody
// typed is the one thing here that cannot be re-fetched.
//
// The transport is `fetch`, not `EventSource`, for one reason: `EventSource`
// cannot send a header, and the capability has to travel in one rather than in
// a URL that reaches the access log and every `Referer` the page sends
// afterwards.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    const PROTOCOL_VERSION = 2;
    const PREFIX = "/model-chain/conversation/v2";
    const HEADER = "x-mc-conversation-key";
    const KEY_FIELD = "mc-llm-chat-conversation-key";

    // The reconnect ladder, in seconds, plus jitter. It stops climbing rather
    // than doubling for ever: a server that has been down for two minutes is
    // not more likely to be up in four, and a page that checks every fifteen
    // seconds costs nothing and comes back the moment it can.
    const BACKOFF = [1000, 2000, 4000, 8000, 15000];

    // How long without a single byte -- events or heartbeats -- before the view
    // says so. The server writes a comment every fifteen seconds, so silence
    // for three of those is the network rather than a quiet conversation.
    const SILENCE = 45000;

    // After this many consecutive stream failures the page stops trying to hold
    // one open and asks for snapshots instead. Polling is transport of last
    // resort, never truth: the cursor still decides what is new.
    const FAILURES_BEFORE_POLLING = 3;
    const POLL_ACTIVE = 2000;
    const POLL_IDLE = 10000;
    const RETRY_STREAM_AFTER = 30000;

    // How often the open stream is asked whether it is still alive. `silent()`
    // could always tell a dead one from a quiet one; nothing ever asked it
    // except a returning tab, so a connection that died without closing -- a
    // laptop lid, a VPN reconnect, a remote-desktop session changing hands --
    // stayed "open" and delivered nothing until the page was reloaded.
    const WATCHDOG = 15000;

    // At a moment the network or the page has just come back, a stream that
    // has been quiet for longer than one heartbeat and some slack is presumed
    // dead rather than waited on for the full SILENCE.
    const STALE = 20000;

    // THE FEED IS OPEN ONLY WHILE SOMETHING IS COMING.
    //
    // It used to be held for the life of the page, and a connection held open
    // while it waits on nothing is the one that came back half-dead -- open as
    // far as the page could tell, and silent -- in every incident that locked
    // the page up. So it opens for a known boundary and closes at its end: a
    // reply being written (from the moment one is asked for until its terminal
    // event), and its read-aloud while it plays. Everything else -- which
    // thread LLM Studio is on, threads made or deleted elsewhere, unread counts
    // -- is read when the panel is opened, when the page comes back and when
    // the workspace changes. See `review`.
    //
    // A page in the background that is still holding a reply keeps it until
    // the reply ends, or for this long at most.
    const HOLD_FOR_REPLY = 600000;

    // The four actions that end in a model being asked for words (the
    // server's GENERATING). Only these anticipate a reply, so only these open
    // the feed before they are sent.
    const GENERATING = ["send", "regenerate", "continue", "resend_from_user"];

    // The Gradio component that carries this process's capability, and where
    // Gradio publishes every component's initial value. A restarted server
    // mints a new key; the page keeps the old one in the DOM for ever, and
    // every request it makes is refused. `/config` is behind Gradio's own
    // login check, so reading the new key from it is exactly as privileged as
    // loading the page was -- which is why this does not add an endpoint of
    // its own that would not be.
    const CONFIG = "/config";
    const RENEW_COOLDOWN = 10000;

    const DRAFT_DEBOUNCE = 250;
    const MAX_CACHED_THREADS = 10;

    function basePath() {
        // Forge can be served under a path by a reverse proxy. The one thing
        // on the page that certainly knows it is the document's own URL, so
        // the prefix is taken from there rather than from a setting somebody
        // has to keep in step.
        const path = (window.location && window.location.pathname) || "/";
        const marker = path.indexOf("/model-chain/");
        if (marker > 0) return path.slice(0, marker);
        const trimmed = path.replace(/\/[^/]*$/, "");
        return trimmed === "/" ? "" : trimmed;
    }

    function route(name) {
        return basePath() + PREFIX + name;
    }

    function storageKey(name) {
        return "forge-assistant:" + (basePath() || "/") + ":" + name;
    }

    // Every storage access is wrapped. A private window, blocked site data, a
    // preview frame and a thumbnail capture all answer differently, and two of
    // them throw rather than returning nothing.
    function readStorage(name) {
        try {
            return window.sessionStorage.getItem(storageKey(name));
        } catch (error) {
            return null;
        }
    }

    function writeStorage(name, value) {
        try {
            window.sessionStorage.setItem(storageKey(name), value);
            return true;
        } catch (error) {
            return false;
        }
    }

    function conversationKey(character, thread) {
        return String(character || "") + "\u0000" + String(thread || "");
    }

    function uuid() {
        const random = window.crypto && window.crypto.getRandomValues
            ? window.crypto.getRandomValues(new Uint8Array(16))
            : null;
        if (!random) {
            return "x" + Date.now().toString(16) + Math.random().toString(16).slice(2, 14);
        }
        let out = "";
        for (let i = 0; i < random.length; i += 1) {
            out += random[i].toString(16).padStart(2, "0");
        }
        return out;
    }

    function Store() {
        this.ready = false;
        this.serverEpoch = "";
        this.pageId = uuid();
        this.capabilities = {};
        this.feed = "";
        this.generation = "";
        this.cursor = 0;
        this.connected = false;
        // Whether a stream has ever been open. "Connecting…" and
        // "Reconnecting…" are different sentences and only one of them is true
        // at a time, and the panel said the second one from the moment it
        // mounted.
        this.everConnected = false;
        this.characters = [];
        this.storageAvailable = true;
        this.failures = 0;
        this.polling = false;
        this.lastTraffic = 0;
        this.selection = {character: "", thread: "", epoch: uuid()};
        this.mode = "";
        this.activeWorkspace = "";
        this.snapshots = new Map();      // conversationKey -> snapshot
        this.order = [];                 // LRU of conversation keys
        this.drafts = new Map();         // conversationKey -> draft
        this.operations = new Map();     // operation id -> operation
        this.unread = new Map();         // conversationKey -> count
        this.listeners = new Set();
        this.speech = {playing: false, owner: "", operation: ""};
        // Whether replies are read aloud: Voice Chat's "Speak replies
        // automatically", as the server last said. `null` until it has said,
        // because "off" is a claim -- see `setReadAloud`.
        this.readAloud = null;
        this.error = "";
        this._pending = null;
        this._draftTimer = null;
        this._reconnect = null;
        this._pollTimer = null;
        this._abort = null;
        this._watchdog = null;
        this._retryStream = null;
        this._renewing = null;
        this._renewedAt = 0;
        this._renewFutile = false;
        this._keyOverride = null;
        this._restarting = false;
        // `asleep` is the page being in the background. `sleeping` is the feed
        // closed on purpose -- which is its resting state: it starts closed
        // and opens only while something is coming. See `review`.
        this.asleep = false;
        this.sleeping = true;
        this._sleepTimer = null;
        this._heldTooLong = false;
        this._expecting = 0;
        this._connecting = null;
        this._restoreDrafts();
    }

    Store.prototype.subscribeState = function (listener) {
        this.listeners.add(listener);
        try {
            listener(this.snapshot());
        } catch (error) {
            console.error("Forge Assistant: a view failed to render", error);
        }
        return () => this.listeners.delete(listener);
    };

    Store.prototype.announce = function () {
        const view = this.snapshot();
        this.listeners.forEach((listener) => {
            try {
                listener(view);
            } catch (error) {
                console.error("Forge Assistant: a view failed to render", error);
            }
        });
    };

    // An immutable-enough view: the arrays and maps views read are rebuilt, so
    // a view that keeps one cannot mutate the store by accident.
    Store.prototype.snapshot = function () {
        const key = conversationKey(this.selection.character, this.selection.thread);
        return {
            ready: this.ready,
            connected: this.connected,
            everConnected: this.everConnected,
            polling: this.polling,
            idle: this.sleeping,
            error: this.error,
            capabilities: Object.assign({}, this.capabilities),
            selection: Object.assign({}, this.selection),
            mode: this.mode,
            activeWorkspace: this.activeWorkspace,
            conversation: this.snapshots.get(key) || null,
            characters: this.characters.slice(),
            draft: Object.assign({text: "", attachment: null, draftVersion: 0,
                                  pendingTranscripts: [], editBuffer: null},
                                 this.drafts.get(key) || {}),
            operation: this.operationFor(key),
            unread: this.unread.get(key) || 0,
            unreadTotal: Array.from(this.unread.values())
                .reduce((total, count) => total + count, 0),
            speech: Object.assign({}, this.speech),
            readAloud: this.readAloud,
            storageAvailable: this.storageAvailable,
        };
    };

    Store.prototype.operationFor = function (key) {
        let found = null;
        this.operations.forEach((operation) => {
            if (operation.key === key && !operation.terminal) found = operation;
        });
        return found ? Object.assign({}, found) : null;
    };

    // -- the capability -------------------------------------------------- //

    function keyInput() {
        const field = document.getElementById(KEY_FIELD);
        if (!field) return null;
        return field.tagName === "TEXTAREA" || field.tagName === "INPUT"
            ? field
            : field.querySelector("textarea, input");
    }

    Store.prototype.key = function () {
        const input = keyInput();
        const found = (input && input.value) || "";
        // A renewed key stands in for the one it replaced and nothing else. If
        // the field ever changes to something else, that is newer than both.
        const override = this._keyOverride;
        if (override && (found === override.stale || !found)) return override.fresh;
        return found;
    };

    /** Ask Gradio for this process's key, when the one on the page is refused.
     *
     * Resolves true when a different key was found and adopted. One request
     * at a time, and not more often than RENEW_COOLDOWN: a server that keeps
     * refusing a fresh key is refusing for a reason this cannot fix, and a
     * page that fetched the whole Gradio config in a loop would be a worse
     * problem than the one it was solving.
     */
    Store.prototype.renewKey = function () {
        if (this._renewing) return this._renewing;
        // Once per episode. A renewal that found the key the page already had
        // means the refusal is about something else, and asking again every
        // ten seconds would be a multi-megabyte config fetched for ever. Any
        // request that succeeds ends the episode.
        if (this._renewFutile) return Promise.resolve(false);
        if (Date.now() - this._renewedAt < RENEW_COOLDOWN) return Promise.resolve(false);
        this._renewedAt = Date.now();
        const stale = this.key();
        this._renewing = fetch(basePath() + CONFIG, {credentials: "same-origin"})
            .then((response) => (response.ok ? response.json() : null))
            .then((config) => {
                const components = (config && config.components) || [];
                const holder = components.find((item) => item && item.props
                    && item.props.elem_id === KEY_FIELD);
                const fresh = holder && typeof holder.props.value === "string"
                    ? holder.props.value : "";
                if (!fresh || fresh === stale) {
                    this._renewFutile = true;
                    return false;
                }
                this._keyOverride = {stale, fresh};
                // Into the field as well, so anything else that reads it is
                // current. No event is dispatched: nothing on the Python side
                // listens to this field, and a synthetic `input` would be a
                // Gradio change nobody asked for.
                const input = keyInput();
                if (input) input.value = fresh;
                return true;
            })
            .catch(() => false)
            .then((renewed) => {
                this._renewing = null;
                return renewed;
            });
        return this._renewing;
    };

    Store.prototype.headers = function (extra) {
        const headers = Object.assign({}, extra || {});
        const key = this.key();
        if (key) headers[HEADER] = key;
        return headers;
    };

    Store.prototype.request = function (path, options) {
        const settings = Object.assign({credentials: "same-origin"}, options || {});
        settings.headers = this.headers(settings.headers);
        return fetch(route(path), settings).then((response) => {
            if (!response.ok) {
                return response.json().catch(() => ({})).then((body) => {
                    const error = new Error((body.error && body.error.message)
                        || ("The server answered " + response.status));
                    error.status = response.status;
                    error.code = body.error && body.error.code;
                    error.body = body;
                    if (response.status === 401) this.refused();
                    throw error;
                });
            }
            this._renewFutile = false;
            return response.json();
        });
    };

    /** A request was refused for its key.
     *
     * That happens for one reason in practice: the server restarted, minted a
     * new key, and this page is still holding the old one. It used to be the
     * end of the conversation until a reload -- "Reload the page", every two
     * seconds, for as long as the tab stayed open. Now the key is renewed and,
     * because a new key means a new process, the session is started over.
     * The refused request still fails; everything it was part of is rebuilt.
     */
    Store.prototype.refused = function () {
        this.renewKey().then((renewed) => {
            if (renewed) this.restarted();
        });
    };

    // -- starting up ------------------------------------------------------ //

    Store.prototype.start = function () {
        if (this._starting) return this._starting;
        this._starting = this.request("/bootstrap?page=" + encodeURIComponent(this.pageId))
            .then((found) => {
                this.ready = true;
                this.serverEpoch = found.server_epoch || "";
                this.capabilities = found.capabilities || {};
                this.characters = found.characters || [];
                this.mode = found.mode || this.mode;
                this.noteReadAloud(found);
                this.error = "";
                // WHICH CONVERSATION. Without this the panel opened on an empty
                // character and an empty thread, `refresh()` returned early
                // because there was nothing to ask about, and the transcript
                // was never going to be given anything to draw.
                //
                // A seed and not a source of truth: the preference it comes
                // from is installation-wide, so it says which conversation was
                // last opened on this machine. This page owns its selection
                // from here (two windows may deliberately differ).
                const seed = found.selection || {};
                if (!this.selection.thread && seed.thread_id) {
                    this.selection = {character: String(seed.character || ""),
                                      thread: String(seed.thread_id || ""),
                                      epoch: uuid()};
                }
                this.announce();
                this.watch();
                // No feed: one is opened only if the snapshot shows a reply
                // on its way. See `review`.
                return this.refresh();
            })
            .catch((error) => {
                this.ready = false;
                // A refused bootstrap is a stale key, and `refused` is already
                // renewing it; saying the host cannot hold a conversation
                // would be wrong for the few hundred milliseconds that takes.
                if (error && error.status === 401) {
                    this.error = "";
                    this.announce();
                    return;
                }
                this.error = "Conversation unavailable on this host version.";
                console.warn("Forge Assistant: could not start a conversation session",
                             error);
                this.announce();
            });
        return this._starting;
    };

    Store.prototype.connect = function () {
        if (this.sleeping) return Promise.resolve();
        return this.request("/subscribe", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({page_id: this.pageId}),
        }).then((found) => {
            this.feed = found.feed || "";
            this.generation = found.subscription_generation || "";
            this.cursor = found.stream_cursor || 0;
            // Resolved here, on the subscription, not when the stream ends: a
            // send waits for this so that the reply it asks for lands on a
            // feed that already exists.
            this.refresh().then(() => this.stream());
        }).catch((error) => {
            console.warn("Forge Assistant: could not open the conversation feed", error);
            this.dropped();
        });
    };

    // -- the stream -------------------------------------------------------- //

    Store.prototype.stream = function () {
        if (!this.feed || this.polling || this.sleeping) return Promise.resolve();
        const controller = typeof AbortController === "function" ? new AbortController()
            : null;
        this._abort = controller;
        const path = "/events?feed=" + encodeURIComponent(this.feed)
            + "&cursor=" + encodeURIComponent(String(this.cursor));
        return fetch(route(path), {
            credentials: "same-origin",
            headers: this.headers(),
            signal: controller ? controller.signal : undefined,
        }).then((response) => {
            if (!response.ok || !response.body) throw new Error("no stream");
            this.connected = true;
            this.everConnected = true;
            this.failures = 0;
            this.lastTraffic = Date.now();
            this.error = "";
            this.announce();
            return this.readStream(response.body.getReader());
        }).catch((error) => {
            if (error && error.name === "AbortError") return;
            console.warn("Forge Assistant: the conversation feed closed", error);
            this.dropped();
        });
    };

    Store.prototype.readStream = function (reader) {
        const decoder = new TextDecoder();
        let buffer = "";
        const pump = () => reader.read().then(({done, value}) => {
            if (done) {
                this.dropped();
                return;
            }
            this.lastTraffic = Date.now();
            buffer += decoder.decode(value, {stream: true});
            let split = buffer.indexOf("\n\n");
            while (split >= 0) {
                this.frame(buffer.slice(0, split));
                buffer = buffer.slice(split + 2);
                split = buffer.indexOf("\n\n");
            }
            return pump();
        });
        return pump();
    };

    Store.prototype.frame = function (raw) {
        const lines = String(raw || "").split("\n");
        let data = "";
        lines.forEach((line) => {
            if (line.indexOf("data:") === 0) data += line.slice(5).trim();
        });
        if (!data) return;                 // a comment: a heartbeat, deliberately
        let event = null;
        try {
            event = JSON.parse(data);
        } catch (error) {
            return;
        }
        this.apply(event);
    };

    Store.prototype.dropped = function () {
        // A feed let go on purpose is not a failure, and nothing climbs the
        // ladder for it: the return opens a new one.
        if (this.sleeping) return;
        if (!this.needsFeed()) {
            // Nothing is coming, so a feed that ended is not a failure to
            // recover from: it stays closed.
            this.letGo();
            this.announce();
            return;
        }
        this.connected = false;
        this.failures += 1;
        this.announce();
        if (this.failures >= FAILURES_BEFORE_POLLING) {
            this.startPolling();
            return;
        }
        const wait = BACKOFF[Math.min(this.failures - 1, BACKOFF.length - 1)]
            + Math.floor(Math.random() * 400);
        window.clearTimeout(this._reconnect);
        this._reconnect = window.setTimeout(() => this.connect(), wait);
    };

    Store.prototype.startPolling = function () {
        if (this.polling) return;
        this.polling = true;
        this.announce();
        const tick = () => {
            if (!this.polling) return;
            const hidden = document.hidden;
            if (hidden) {
                this._pollTimer = window.setTimeout(tick, POLL_IDLE);
                return;
            }
            this.refresh().finally(() => {
                this._pollTimer = window.setTimeout(tick, POLL_ACTIVE);
            });
        };
        this._pollTimer = window.setTimeout(tick, POLL_ACTIVE);
        window.clearTimeout(this._retryStream);
        this._retryStream = window.setTimeout(() => {
            // Try the stream again, without giving up the fallback until one
            // actually opens. A page that stopped polling to try a stream that
            // fails again is a page that showed nothing for thirty seconds.
            this.polling = false;
            this.failures = 0;
            window.clearTimeout(this._pollTimer);
            this.connect();
        }, RETRY_STREAM_AFTER);
    };

    Store.prototype.silent = function (limit) {
        return !!(this.connected && this.lastTraffic
            && Date.now() - this.lastTraffic > (limit || SILENCE));
    };

    /** Tear the current stream down and go through the reconnect ladder.
     *
     * The abort is what makes this different from `dropped` on its own: a
     * stream that died without closing is a `reader.read()` that will never
     * settle, and nothing short of aborting the fetch lets go of it. The
     * AbortError that follows is swallowed by `stream`, so the drop is counted
     * once, here.
     */
    Store.prototype.restream = function () {
        this.connected = false;
        if (this._abort) {
            try {
                this._abort.abort();
            } catch (error) { /* already gone */ }
            this._abort = null;
        }
        this.dropped();
    };

    // One timer, every fifteen seconds, doing one subtraction. It re-arms
    // itself for the life of the page; `dispose` is what stops it.
    Store.prototype.watch = function () {
        window.clearTimeout(this._watchdog);
        this._watchdog = window.setTimeout(() => {
            if (this.silent()) this.restream();
            this.watch();
        }, WATCHDOG);
    };

    /** The process behind this page is a different one now.
     *
     * Nothing the page holds about the old one means anything to the new one
     * -- not the feed, not the cursor, not an operation id -- and the old
     * answer to that was a sentence: "Reload the page to carry on". This is
     * the reload without the page: the session is started again from the
     * bootstrap. Drafts survive it, because they were never the server's.
     *
     * A send that was in flight is *not* retried. The old process may have
     * written it before it went, and a retry into a registry that has never
     * heard of it is how one message becomes two. Its text is still in the
     * draft, because a draft is only cleared by an acknowledged send, and the
     * conversation as it comes back shows whether it arrived.
     */
    Store.prototype.restarted = function () {
        if (this._restarting) return this._starting || Promise.resolve();
        this._restarting = true;
        window.clearTimeout(this._reconnect);
        window.clearTimeout(this._pollTimer);
        window.clearTimeout(this._retryStream);
        if (this._abort) {
            try {
                this._abort.abort();
            } catch (error) { /* already gone */ }
            this._abort = null;
        }
        this.serverEpoch = "";
        this.feed = "";
        this.generation = "";
        this.cursor = 0;
        this.connected = false;
        this.polling = false;
        this.failures = 0;
        this.operations.clear();
        this._pending = null;
        this.error = "";
        this._starting = null;
        this.announce();
        return this.start().finally(() => {
            this._restarting = false;
        });
    };

    // -- the reducer ------------------------------------------------------- //

    Store.prototype.apply = function (event) {
        if (!event || event.protocol_version !== PROTOCOL_VERSION) return false;
        if (this.serverEpoch && event.server_epoch !== this.serverEpoch) {
            // The process restarted. Nothing this page holds means anything to
            // it -- not cursors, not operation ids -- so none of it is
            // reconciled: the session is started again. See `restarted`.
            this.restarted();
            return false;
        }
        const cursor = Number(event.stream_cursor || 0);
        if (cursor && cursor <= this.cursor) return false;   // duplicate or older
        if (cursor && this.cursor && cursor > this.cursor + 1) {
            // A gap: something was trimmed out of the ring before this page
            // read it. Nothing is guessed -- the whole conversation is asked
            // for again.
            this.cursor = cursor;
            this.refresh();
            return true;
        }
        if (cursor) this.cursor = cursor;
        this.lastTraffic = Date.now();

        const where = event.conversation || {};
        const key = conversationKey(where.character, where.thread_id);
        const mine = key === conversationKey(this.selection.character,
                                             this.selection.thread);

        switch (event.kind) {
        case "mode_changed":
            this.mode = (event.payload && event.payload.mode) || "";
            break;
        case "character_changed":
            // The tab moved to another conversation. Following it is the whole
            // point of being a second window onto the same work -- a panel that
            // stayed on the thread it was seeded with would be a panel showing
            // a different conversation from the one behind it, with nothing on
            // screen to say so.
            this.follow(event.payload || {});
            break;
        case "reply_patch":
            this.patch(event, key, mine);
            break;
        case "operation_accepted":
        case "operation_phase":
            this.phase(event, key);
            break;
        case "operation_terminal":
            this.phase(event, key, true);
            if (!mine || !this.atBottom) this.mark(key);
            if (mine) this.refresh();
            break;
        case "result_conflict":
            this.phase(event, key, true);
            if (mine) this.refresh();
            break;
        case "thread_deleted":
            this.snapshots.delete(key);
            if (mine) this.error = "This conversation was deleted.";
            break;
        case "speech_state":
            this.speech = {
                playing: !!(event.payload && event.payload.playing),
                owner: (event.payload && event.payload.owner) || "",
                operation: event.operation_id || "",
            };
            break;
        case "conversation_changed":
        case "thread_created":
        case "capabilities_changed":
            if (mine) this.refresh();
            else this.snapshots.delete(key);
            break;
        case "reset_required":
            this.refresh();
            break;
        default:
            break;
        }
        this.announce();
        // A reply ending, or its read-aloud stopping, is the end of the
        // boundary the feed was opened for.
        if (event.kind === "operation_terminal" || event.kind === "result_conflict"
            || event.kind === "speech_state") {
            this.review();
        }
        return true;
    };

    Store.prototype.patch = function (event, key, mine) {
        const operation = this.operations.get(event.operation_id) || {
            id: event.operation_id, key, terminal: false, seq: -1,
        };
        const seq = Number(event.operation_seq || 0);
        if (seq && operation.seq && seq <= operation.seq) return;   // obsolete
        operation.seq = seq;
        operation.text = (event.payload && event.payload.text) || "";
        operation.targetIndex = event.payload && event.payload.target_index;
        operation.phase = (event.payload && event.payload.phase) || "generating";
        operation.provisional = true;
        this.operations.set(event.operation_id, operation);
        if (!mine) this.mark(key, 0);    // a badge, never the other thread's view
    };

    Store.prototype.phase = function (event, key, terminal) {
        const operation = this.operations.get(event.operation_id) || {
            id: event.operation_id, key, seq: -1,
        };
        const seq = Number(event.operation_seq || 0);
        if (seq && operation.seq && seq < operation.seq) return;
        operation.seq = seq;
        operation.key = key;
        operation.phase = (event.payload && event.payload.phase) || operation.phase;
        operation.status = (event.payload && event.payload.status) || operation.status;
        operation.text = (event.payload && event.payload.text) || operation.text || "";
        operation.error = (event.payload && event.payload.error) || "";
        operation.terminal = !!terminal;
        operation.provisional = !terminal;
        this.operations.set(event.operation_id, operation);
        if (terminal && this._pending && this._pending.operationId === event.operation_id) {
            this._pending = null;
        }
    };

    Store.prototype.mark = function (key, by) {
        const count = this.unread.get(key) || 0;
        this.unread.set(key, count + (by === undefined ? 1 : by));
    };

    Store.prototype.clearUnread = function (key) {
        this.unread.delete(key || conversationKey(this.selection.character,
                                                  this.selection.thread));
        this.announce();
    };

    // -- snapshots --------------------------------------------------------- //

    Store.prototype.refresh = function () {
        const character = this.selection.character;
        const thread = this.selection.thread;
        const epoch = this.selection.epoch;
        if (!character || !thread) return Promise.resolve(null);
        const path = "/snapshot?character=" + encodeURIComponent(character)
            + "&thread=" + encodeURIComponent(thread)
            + "&selection_epoch=" + encodeURIComponent(epoch);
        return this.request(path).then((found) => {
            // A late answer for a selection the reader has moved away from
            // updates the cache and nothing else. It must never redraw the
            // thread they are looking at now.
            const key = conversationKey(character, thread);
            this.remember(key, found);
            this.noteOperation(key, found);
            if (epoch === this.selection.epoch) {
                this.error = (found.error && found.error.message) || "";
                this.announce();
            }
            this.review();
            return found;
        }).catch((error) => {
            console.warn("Forge Assistant: could not read the conversation", error);
            return null;
        });
    };

    Store.prototype.remember = function (key, snapshot) {
        this.snapshots.set(key, snapshot);
        this.order = this.order.filter((item) => item !== key);
        this.order.push(key);
        // A cache is allowed to forget a conversation. It is never allowed to
        // forget what somebody typed into one, so a thread with a draft is
        // skipped rather than evicted -- and skipped by *walking past it*
        // rather than by putting it back, which is the shape that spins for
        // ever the first time the oldest entry is one with a draft in it.
        let index = 0;
        while (this.order.length > MAX_CACHED_THREADS && index < this.order.length) {
            const oldest = this.order[index];
            if (this.drafts.has(oldest)) {
                index += 1;
                continue;
            }
            this.order.splice(index, 1);
            this.snapshots.delete(oldest);
        }
    };

    Store.prototype.follow = function (payload) {
        const character = String(payload.character || "");
        const thread = String(payload.thread_id || "");
        if (!thread) return Promise.resolve(null);
        if (character === this.selection.character && thread === this.selection.thread) {
            return Promise.resolve(null);
        }
        return this.select(character, thread);
    };

    Store.prototype.select = function (character, thread) {
        this.flushDrafts();
        this.selection = {character: String(character || ""), thread: String(thread || ""),
                          epoch: uuid()};
        this.clearUnread();
        this.announce();
        return this.refresh();
    };

    // -- drafts ------------------------------------------------------------ //

    Store.prototype.draft = function (key) {
        const wanted = key || conversationKey(this.selection.character,
                                              this.selection.thread);
        if (!this.drafts.has(wanted)) {
            this.drafts.set(wanted, {text: "", attachment: null, draftVersion: 0,
                                     pendingTranscripts: [], editBuffer: null});
        }
        return this.drafts.get(wanted);
    };

    Store.prototype.setDraftText = function (text, key) {
        const draft = this.draft(key);
        draft.text = String(text === undefined || text === null ? "" : text);
        draft.draftVersion += 1;
        this.scheduleDraftSave();
        this.announce();
        return draft;
    };

    Store.prototype.setAttachment = function (attachment, key) {
        const draft = this.draft(key);
        if (draft.attachment && draft.attachment.preview
            && draft.attachment.preview !== (attachment && attachment.preview)) {
            try {
                URL.revokeObjectURL(draft.attachment.preview);
            } catch (error) { /* a preview that was never an object URL */ }
        }
        draft.attachment = attachment || null;
        this.announce();
        return draft;
    };

    Store.prototype.scheduleDraftSave = function () {
        window.clearTimeout(this._draftTimer);
        this._draftTimer = window.setTimeout(() => this.flushDrafts(), DRAFT_DEBOUNCE);
    };

    Store.prototype.flushDrafts = function () {
        window.clearTimeout(this._draftTimer);
        const out = {};
        this.drafts.forEach((draft, key) => {
            // Text only. An attachment is bytes, a transcript may be audio, and
            // neither belongs in storage a second page can read.
            if (draft.text) out[key] = draft.text;
        });
        this.storageAvailable = writeStorage("drafts:v5", JSON.stringify(out));
    };

    Store.prototype._restoreDrafts = function () {
        const raw = readStorage("drafts:v5");
        if (!raw) return;
        try {
            const found = JSON.parse(raw);
            Object.keys(found || {}).forEach((key) => {
                this.drafts.set(key, {text: String(found[key] || ""), attachment: null,
                                      draftVersion: 0, pendingTranscripts: [],
                                      editBuffer: null});
            });
        } catch (error) {
            // Malformed storage is ignored rather than cleared: it is somebody's
            // words, and a reader that cannot parse them should not delete them.
        }
    };

    // -- reading aloud ----------------------------------------------------- //
    //
    // One setting, Voice Chat's "Speak replies automatically", shared with the
    // tab. The flyout's switch used to be a button that started "off" whatever
    // that setting said, never asked, and changed it by pressing the tab's
    // checkbox from here -- so a first press often turned on something that
    // was already on, and replies went on being synthesised while the switch
    // said otherwise. Now the server says what it is (the bootstrap), and the
    // server is what is told (`setReadAloud`).

    /** Take the setting from a bootstrap, when it carries one. */
    Store.prototype.noteReadAloud = function (found) {
        if (found && typeof found.read_aloud === "boolean") {
            this.readAloud = found.read_aloud;
        }
    };

    /** Turn reading aloud on or off. Resolves with what the server stored.
     *
     * Drawn at once, because a switch that waits for a round trip before it
     * moves reads as a switch that did not hear the press -- and put back if
     * the server refuses, because a switch left showing a state the server
     * does not hold is the defect this replaces.
     */
    Store.prototype.setReadAloud = function (on) {
        const before = this.readAloud;
        this.readAloud = !!on;
        this.announce();
        return this.request("/read-aloud", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({read_aloud: !!on}),
        }).then((found) => {
            this.noteReadAloud(found);
            this.announce();
            return this.readAloud;
        }).catch((error) => {
            this.readAloud = before;
            this.announce();
            throw error;
        });
    };

    /** Say, on a command that asks for a reply, whether it may be spoken.
     *
     * Off is `voice: false`, and the server then starts no speech for that
     * reply at all -- no voice engine warmed, nothing synthesised -- whatever
     * the shared setting says. That is what makes the switch mean off even if
     * writing the setting failed.
     *
     * Written once per envelope and never rewritten: a retry sends the same
     * envelope, and a payload that changed between two sends of one operation
     * id is refused by the server as a different request.
     */
    Store.prototype.stampVoice = function (envelope) {
        const payload = envelope.payload || (envelope.payload = {});
        if (Object.prototype.hasOwnProperty.call(payload, "voice")) return;
        payload.voice = this.readAloud !== false;
    };

    // -- commands ---------------------------------------------------------- //

    Store.prototype.expected = function () {
        const conversation = this.snapshot().conversation;
        const revision = conversation && conversation.conversation
            && conversation.conversation.revision;
        if (revision === undefined || revision === null) return null;
        if (typeof revision === "number") return {kind: "revision", value: revision};
        return {kind: "legacy", fingerprint: String(revision)};
    };

    Store.prototype.envelope = function (action, extra) {
        return Object.assign({
            protocol_version: PROTOCOL_VERSION,
            server_epoch: this.serverEpoch,
            operation_id: uuid(),
            issued_at: new Date().toISOString(),
            page_id: this.pageId,
            conversation: {character: this.selection.character,
                           thread_id: this.selection.thread},
            expected_revision: this.expected(),
            selection_epoch: this.selection.epoch,
            payload: {},
        }, extra || {}, {action});
    };

    Store.prototype.send = function (envelope) {
        // A reply is being asked for: the feed opens before the request goes,
        // so the reply lands on a feed that is already there. Anything else is
        // sent with no feed at all.
        const generating = !!envelope && GENERATING.indexOf(envelope.action) >= 0;
        if (generating) this.stampVoice(envelope);
        if (generating) this._expecting += 1;
        const settle = () => {
            if (!generating) return;
            this._expecting = Math.max(0, this._expecting - 1);
            this.review();
        };
        const ready = generating ? this.ensureFeed() : Promise.resolve();
        return ready.then(() => this.request("/commands", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(envelope),
        })).then((outcome) => {
            this.after(outcome);
            settle();
            return outcome;
        }).catch((error) => {
            settle();
            if (error && error.body) {
                this.after(error.body);
                return error.body;
            }
            // The response was lost, not refused. The envelope is kept so the
            // same id can be retried, and the page asks what happened rather
            // than guessing -- guessing is how one message becomes two.
            return {ok: false, lost: true, envelope,
                    error: {code: "UNKNOWN",
                            message: "Checking whether your message was sent…",
                            retryable: true}};
        });
    };

    Store.prototype.after = function (outcome) {
        if (!outcome || !outcome.ok) {
            this.announce();
            return;
        }
        const where = outcome.resulting_conversation || {};
        const key = conversationKey(where.character, where.thread_id);
        if (outcome.operation_id && outcome.phase && outcome.phase !== "completed") {
            this.operations.set(outcome.operation_id, {
                id: outcome.operation_id, key, phase: outcome.phase, seq: 0,
                text: "", terminal: false, provisional: true,
                targetIndex: outcome.target && outcome.target.index,
            });
        }
        this.refresh();
    };

    // The one submission that has to be idempotent under a shaking hand: Send.
    // The id is latched with the text, so a second click, a second Enter, and a
    // retry after a lost response are all the same request.
    Store.prototype.submit = function (extra) {
        const key = conversationKey(this.selection.character, this.selection.thread);
        const draft = this.draft(key);
        if (this._pending && !this._pending.settled) {
            return Promise.resolve({ok: true, duplicate: true,
                                    operation_id: this._pending.operationId});
        }
        const envelope = this.envelope("send", extra);
        envelope.payload = Object.assign({text: draft.text}, envelope.payload || {},
                                         (extra && extra.payload) || {});
        if (draft.attachment && draft.attachment.token && draft.attachment.state === "ready") {
            envelope.payload.attachment_token = draft.attachment.token;
        }
        const latched = {operationId: envelope.operation_id, key,
                         draftVersion: draft.draftVersion, envelope, settled: false};
        this._pending = latched;
        return this.send(envelope).then((outcome) => {
            latched.settled = !!(outcome && (outcome.ok || !outcome.lost));
            if (outcome && outcome.ok) {
                const current = this.draft(key);
                // Only if nothing newer was typed while this was in flight.
                if (current.draftVersion === latched.draftVersion) {
                    current.text = "";
                    current.attachment = null;
                    current.draftVersion += 1;
                    this.scheduleDraftSave();
                }
                this._pending = null;
            }
            this.announce();
            return outcome;
        });
    };

    Store.prototype.retryPending = function () {
        if (!this._pending) return Promise.resolve(null);
        // The SAME envelope, and therefore the same operation id. A new id here
        // would be a second message.
        return this.send(this._pending.envelope);
    };

    Store.prototype.checkPending = function () {
        if (!this._pending) return Promise.resolve(null);
        return this.request("/operations/" + encodeURIComponent(this._pending.operationId))
            .catch(() => null);
    };

    Store.prototype.stop = function (operationId) {
        const envelope = this.envelope("stop", {expected_revision: null});
        envelope.payload = {operation_id: operationId};
        return this.send(envelope);
    };

    Store.prototype.upload = function (file) {
        const form = new FormData();
        form.append("file", file, file.name || "pasted-image");
        return fetch(route("/attachments"), {
            method: "POST", credentials: "same-origin",
            headers: this.headers(), body: form,
        }).then((response) => response.json().then((body) => {
            if (!response.ok) throw Object.assign(new Error(body.reason
                || "That picture could not be used."), {body});
            return body;
        }));
    };

    Store.prototype.resolveResult = function (operationId, decision) {
        return this.request("/operations/" + encodeURIComponent(operationId) + "/resolve", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({decision}),
        });
    };

    /** Check the session at a moment something may have changed under it.
     *
     * `eager` is for the moments the network or the page has just come back
     * -- `online`, a page restored from the back-forward cache, a tab the
     * browser froze and resumed. The stream is presumed dead after one missed
     * heartbeat rather than three, because at those moments it usually is.
     */
    Store.prototype.reconcile = function (eager) {
        if (this.sleeping) return this.check();
        if (this.silent(eager ? STALE : SILENCE)) this.restream();
        return this.refresh();
    };

    /** Whether anything is coming that the feed is for: a reply being asked
     * for, a reply being written, or a reply being read aloud. */
    Store.prototype.needsFeed = function () {
        return this._expecting > 0 || this.replying()
            || !!(this.speech && this.speech.playing);
    };

    /** Open the feed if something is coming, close it if nothing is. Called
     * after every moment that could change the answer: a snapshot, a send, a
     * reply ending, speech stopping, the page going away or coming back. */
    Store.prototype.review = function () {
        if (!this.ready) return;
        if (this.needsFeed()) {
            // A page in the background that already held a reply for as long
            // as it may does not open another feed for it until it is back.
            if (this.asleep && this._heldTooLong) return;
            this.ensureFeed();
            return;
        }
        if (!this.sleeping) this.letGo();
    };

    /** The feed, opened if it is closed. Resolves once the subscription
     * exists, which is what a send waits for. */
    Store.prototype.ensureFeed = function () {
        if (!this.sleeping) return this._connecting || Promise.resolve();
        this.sleeping = false;
        this._connecting = this.connect().then(() => { this._connecting = null; });
        return this._connecting;
    };

    /** What a snapshot says about the reply in this conversation. One in
     * flight that this page did not know about -- started in LLM Studio, or
     * in another window -- is taken in, so the feed opens for it; one this
     * page thought was still running but the server has finished is marked
     * finished, so a missed terminal event cannot hold the feed open. */
    Store.prototype.noteOperation = function (key, found) {
        const running = found && found.operation;
        if (running && running.operation_id) {
            const known = this.operations.get(running.operation_id);
            if (!known) {
                this.operations.set(running.operation_id, {
                    id: running.operation_id, key, seq: -1, terminal: false, provisional: true,
                    phase: running.status || "", status: running.status || "",
                    text: running.final_text || "",
                });
            }
            return;
        }
        if (!found || found.error) return;
        this.operations.forEach((operation) => {
            if (operation && operation.key === key && !operation.terminal
                && !(this._pending && this._pending.operationId === operation.id)) {
                operation.terminal = true;
                operation.provisional = false;
            }
        });
    };

    /** The page went to the background. Nothing is coming: the feed is
     * already closed. A reply is coming: it is held until it ends, or for
     * HOLD_FOR_REPLY at most. */
    Store.prototype.sleep = function () {
        this.asleep = true;
        if (this.sleeping) return;
        if (!this.needsFeed()) {
            this.letGo();
            return;
        }
        window.clearTimeout(this._sleepTimer);
        this._sleepTimer = window.setTimeout(() => {
            if (!this.asleep) return;
            this._heldTooLong = true;
            this.letGo();
        }, HOLD_FOR_REPLY);
    };

    /** Whether a reply is in flight: one this page sent, or one it is being
     * told about. */
    Store.prototype.replying = function () {
        if (this._pending) return true;
        let live = false;
        this.operations.forEach((operation) => {
            if (operation && !operation.terminal) live = true;
        });
        return live;
    };

    Store.prototype.letGo = function () {
        window.clearTimeout(this._sleepTimer);
        this._sleepTimer = null;
        this.sleeping = true;
        this.connected = false;
        this.failures = 0;
        this.polling = false;
        this._connecting = null;
        window.clearTimeout(this._reconnect);
        window.clearTimeout(this._pollTimer);
        window.clearTimeout(this._retryStream);
        if (this._abort) {
            try {
                this._abort.abort();
            } catch (error) { /* already gone */ }
            this._abort = null;
        }
    };

    /** Back on screen. A feed that is open is checked the way any return
     * checks it; with none open, the conversation is read, and a feed opened
     * only if the snapshot shows a reply on its way. */
    Store.prototype.wake = function () {
        this.asleep = false;
        this._heldTooLong = false;
        window.clearTimeout(this._sleepTimer);
        this._sleepTimer = null;
        if (!this.sleeping) return this.reconcile(false);
        return this.check();
    };

    /** One look at what changed while nothing was open: which conversation
     * LLM Studio is on (followed, as the feed's character_changed was), the
     * characters and capabilities, and the conversation itself -- whose
     * snapshot decides whether a feed is needed. The moments for it are the
     * panel opening, the page coming back and the workspace changing. */
    Store.prototype.check = function () {
        if (!this.ready || this.asleep) return Promise.resolve(null);
        return this.request("/bootstrap?page=" + encodeURIComponent(this.pageId))
            .then((found) => {
                this.capabilities = found.capabilities || this.capabilities;
                this.characters = found.characters || this.characters;
                this.mode = found.mode || this.mode;
                this.noteReadAloud(found);
                const seed = found.selection || {};
                if (seed.thread_id && (String(seed.character || "") !== this.selection.character
                                       || String(seed.thread_id) !== this.selection.thread)) {
                    return this.follow(seed);
                }
                return this.refresh();
            })
            .catch(() => this.refresh());
    };

    Store.prototype.dispose = function () {
        window.clearTimeout(this._sleepTimer);
        window.clearTimeout(this._reconnect);
        window.clearTimeout(this._pollTimer);
        window.clearTimeout(this._draftTimer);
        window.clearTimeout(this._watchdog);
        window.clearTimeout(this._retryStream);
        this.flushDrafts();
        this.polling = false;
        if (this._abort) {
            try {
                this._abort.abort();
            } catch (error) { /* already gone */ }
        }
        this.listeners.clear();
    };

    NS.Store = Store;
    NS.conversationKey = conversationKey;
    NS.basePath = basePath;
    NS.route = route;
    NS.uuid = uuid;

    NS.store = function () {
        if (!NS._store) NS._store = new Store();
        return NS._store;
    };
})();
