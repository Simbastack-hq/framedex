// fdx share editor: the gallery itself is the workspace. Click a photo to
// view it in the visitors' viewer; Hide is explicit (the button on the photo,
// the one in the viewer, or H). Drag (or Alt+arrows) to reorder within a day,
// click the title to rename, pick the cover, set a passcode, Publish. Every
// change is saved to selection.txt by the local server, which returns the
// re-rendered page.
(function () {
  var d = document, fdx = window.fdx;
  var cfg = JSON.parse(d.getElementById("fdx-edit").textContent);
  var g = d.getElementById("g"), bar = d.querySelector(".bar");
  var toast = d.querySelector(".toast"), note = toast.querySelector(".note");
  var undo = toast.querySelector(".undo"), hint = bar.querySelector(".hint");
  var hid = bar.querySelector(".hid"), toastTimer = null, undoRel = null;
  var act = bar.querySelector(".act"), done = bar.querySelector(".done");
  var lockon = bar.querySelector(".lockon"), pass = bar.querySelector(".pass");
  var code = bar.querySelector(".code"), show = bar.querySelector(".show");
  var go = bar.querySelector(".go"), link = bar.querySelector(".link"), copy = bar.querySelector(".copy");
  var cover = cfg.cover, queue = Promise.resolve(), dragRel = null, published = false;
  var unsaved = false; // the last change did not reach selection.txt: Publish waits
  var down = null, dragEnd = 0, DRAG_PX = 6; // a press that moved is a drag, not a click
  var vb = null; // the viewer's Hide / Restore button
  var keys = false; // the last input was the keyboard: keep focus on the photo across saves
  d.addEventListener("keydown", function () { keys = true; }, true);
  d.addEventListener("pointerdown", function () { keys = false; }, true);

  function say(text, err, withUndo) {
    clearTimeout(toastTimer);
    note.textContent = text || ""; toast.classList.toggle("err", !!err);
    undo.hidden = !withUndo; toast.hidden = !text;
    if (text && !err) toastTimer = setTimeout(function () { toast.hidden = true; undoRel = null; }, 5000);
  }
  undo.addEventListener("click", function () {
    var rel = undoRel; undoRel = null; say("");
    if (rel && q(rel)) { q(rel).classList.remove("off"); save(rel); }
  });
  function q(rel) { return g.querySelector('[data-rel="' + CSS.escape(rel) + '"]'); }
  function post(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-FDX-Token": cfg.token },
      body: JSON.stringify(body)
    }).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; });
    });
  }
  function model() {
    return [].map.call(g.querySelectorAll("main > section"), function (s) {
      return {
        items: [].map.call(s.querySelectorAll("[data-rel]"), function (el) {
          return { rel: el.dataset.rel, hidden: el.classList.contains("off") };
        })
      };
    });
  }
  function title() { return g.querySelector(".top h1").textContent; }

  function save(anchor, edit) {
    var before = anchor && q(anchor) ? q(anchor).getBoundingClientRect().top : null;
    var focused = keys && d.activeElement.closest && d.activeElement.closest(".ph[data-rel]");
    var refocus = focused ? focused.dataset.rel : null;
    var payload = { title: title(), cover: cover, sections: model() };
    if (edit) edit(payload);
    var done = queue.then(function () { return post("api/save", payload); }).then(function (j) {
      unsaved = false;
      g.innerHTML = j.html; wire(); d.title = "Editing \u00b7 " + title();
      hid.textContent = j.hidden ? " " + j.hidden + " hidden." : "";
      if (before !== null && q(anchor)) window.scrollBy(0, q(anchor).getBoundingClientRect().top - before);
      if (refocus && !fdx.current() && q(refocus)) q(refocus).querySelector("a").focus({ preventScroll: true });
      hint.hidden = true; changed();
      return true;
    }, function (e) {
      unsaved = true;
      say("Couldn’t save that change (" + e.message + "). Make it again before publishing.", true);
      return false;
    });
    queue = done;
    return done;
  }
  function changed() {
    if (!published) return;
    done.hidden = true; act.hidden = false; go.textContent = "Publish changes";
  }

  function toggle(rel) {
    var fig = q(rel);
    if (!fig || fig.closest(".hero")) return; // the cover can't be hidden; Remove cover first
    var hiding = !fig.classList.contains("off");
    fig.classList.toggle("off"); label();
    save(rel).then(function (ok) {
      // in the viewer its own button reads "Hidden · Restore": no toast on top of it
      if (ok && hiding && !fdx.current()) { undoRel = rel; say("Photo hidden.", false, true); }
    });
  }
  function label(shown) {
    var a = shown || fdx.current(), fig = a && a.closest(".ph[data-rel]");
    if (!vb || !fig) return;
    vb.hidden = !!fig.closest(".hero");
    vb.textContent = fig.classList.contains("off") ? "Hidden \u00b7 Restore" : "Hide";
  }

  function move(rel, delta) {
    var sec = model().filter(function (s) { return s.items.some(function (i) { return i.rel === rel; }); })[0];
    if (!sec) return;
    var at = sec.items.findIndex(function (i) { return i.rel === rel; }), to = at + delta;
    if (to < 0 || to >= sec.items.length) return;
    save(rel, function (p) {
      p.sections.forEach(function (s) {
        var k = s.items.findIndex(function (i) { return i.rel === rel; });
        if (k < 0) return;
        var it = s.items.splice(k, 1)[0]; s.items.splice(to, 0, it);
      });
    });
  }
  function drop(rel, target, after) {
    save(rel, function (p) {
      p.sections.forEach(function (s) {
        var k = s.items.findIndex(function (i) { return i.rel === rel; });
        var t = s.items.findIndex(function (i) { return i.rel === target; });
        if (k < 0 || t < 0) return;
        var it = s.items.splice(k, 1)[0];
        t = s.items.findIndex(function (i) { return i.rel === target; });
        s.items.splice(after ? t + 1 : t, 0, it);
      });
    });
  }
  function section(el) { return el.closest("section"); }
  function clearMarks() {
    [].forEach.call(g.querySelectorAll(".drop-before,.drop-after"), function (el) {
      el.classList.remove("drop-before", "drop-after");
    });
  }

  function wire() {
    fdx.reveal(g); fdx.bind(g); label();
    var h1 = g.querySelector(".top h1"), was = h1.textContent;
    h1.setAttribute("spellcheck", "false");
    h1.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); h1.blur(); }
      else if (e.key === "Escape") { h1.textContent = was; h1.blur(); }
    });
    h1.addEventListener("blur", function () {
      var now = h1.textContent.replace(/\s+/g, " ").trim();
      if (!now) { h1.textContent = was; return; }
      if (now !== was) save(null);
    });
    [].forEach.call(g.querySelectorAll(".ph[data-rel]"), function (fig) {
      var rel = fig.dataset.rel, a = fig.querySelector("a"), hero = !!fig.closest(".hero");
      var off = fig.classList.contains("off");
      // Overlays sit on the photo itself, above any caption.
      var frame = d.createElement("div"); frame.className = "frame";
      a.parentNode.insertBefore(frame, a); frame.appendChild(a);
      var tools = d.createElement("div"); tools.className = "tools";
      if (hero) {
        tools.innerHTML = '<button type="button" class="t-uncover">Remove cover</button>';
        var tag = d.createElement("span"); tag.className = "cover-tag"; tag.textContent = "Cover";
        frame.appendChild(tag);
      } else if (!off) {
        tools.innerHTML = '<button type="button" class="t-cover">Make cover</button>';
      }
      frame.appendChild(tools);
      if (off) {
        var b = d.createElement("button"); b.type = "button"; b.className = "badge";
        b.textContent = "Hidden \u00b7 Restore"; frame.appendChild(b);
        a.setAttribute("aria-label", "Hidden: " + (a.querySelector("img").alt || "photo"));
      } else if (!hero) {
        var h = d.createElement("button"); h.type = "button"; h.className = "t-hide";
        h.textContent = "Hide"; h.title = "Hide (H)"; frame.appendChild(h);
      }
      frame.addEventListener("click", function (e) {
        var t = e.target.classList;
        if (t.contains("t-cover")) { cover = rel; save(rel); }
        else if (t.contains("t-uncover")) { cover = "none"; save(null); }
        else if (t.contains("t-hide") || t.contains("badge")) toggle(rel);
      });
      fig.addEventListener("pointerdown", function (e) { down = { x: e.clientX, y: e.clientY }; });
      a.addEventListener("click", function (e) {
        if (e.metaKey || e.ctrlKey || e.shiftKey) return; // the browser's own new-tab click
        e.preventDefault();
        var moved = down && e.detail && Math.hypot(e.clientX - down.x, e.clientY - down.y) > DRAG_PX;
        down = null;
        if (moved || Date.now() - dragEnd < 300) return; // that was a drag
        fdx.open(a);
      });
      a.addEventListener("keydown", function (e) {
        if (hero) return;
        if (e.altKey && (e.key === "ArrowUp" || e.key === "ArrowLeft")) { e.preventDefault(); move(rel, -1); }
        else if (e.altKey && (e.key === "ArrowDown" || e.key === "ArrowRight")) { e.preventDefault(); move(rel, 1); }
      });
      if (hero) return;
      fig.draggable = true;
      fig.addEventListener("dragstart", function (e) {
        dragRel = rel; fig.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", rel);
      });
      fig.addEventListener("dragend", function () {
        fig.classList.remove("dragging"); clearMarks(); dragRel = null; dragEnd = Date.now();
      });
      fig.addEventListener("dragover", function (e) {
        if (!dragRel || dragRel === rel) return;
        var from = q(dragRel);
        if (!from || section(from) !== section(fig)) return;
        e.preventDefault(); clearMarks();
        var r = fig.getBoundingClientRect(), sideways = !!fig.closest(".pair") && innerWidth > 600;
        var after = sideways ? e.clientX > r.left + r.width / 2 : e.clientY > r.top + r.height / 2;
        fig.classList.add(after ? "drop-after" : "drop-before");
      });
      fig.addEventListener("drop", function (e) {
        e.preventDefault();
        var after = fig.classList.contains("drop-after"); clearMarks();
        if (dragRel && dragRel !== rel) drop(dragRel, rel, after);
      });
    });
  }
  // H hides or restores the photo in the viewer, else the focused photo.
  d.addEventListener("keydown", function (e) {
    if ((e.key !== "h" && e.key !== "H") || e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target;
    if (t.isContentEditable || t.tagName === "INPUT" || t.tagName === "TEXTAREA") return;
    var a = fdx.current(), fig = a ? a.closest(".ph[data-rel]") : t.closest && t.closest(".ph[data-rel]");
    if (fig) { e.preventDefault(); toggle(fig.dataset.rel); }
  });
  d.addEventListener("dragover", function (e) {
    if (!dragRel) return;
    if (e.clientY < 90) scrollBy(0, -18); else if (e.clientY > innerHeight - 90) scrollBy(0, 18);
  });

  function lockUi() {
    pass.hidden = !lockon.checked;
    if (lockon.checked) code.focus();
  }
  lockon.addEventListener("change", function () { lockUi(); say(""); changed(); });
  show.addEventListener("click", function () {
    var on = code.type === "password"; code.type = on ? "text" : "password"; show.textContent = on ? "Hide" : "Show";
  });
  go.addEventListener("click", function () {
    if (!cfg.configured) {
      say("Publishing isn’t set up yet: add ~/.config/framedex/share.yaml (see docs/share.md).", true);
      return;
    }
    if (lockon.checked && code.value.length < 4) { say("Choose a passcode of at least 4 characters.", true); code.focus(); return; }
    var body = { passcode: lockon.checked ? code.value : null, open: !lockon.checked };
    go.disabled = true; go.textContent = "Publishing…"; say("");
    queue.then(function () {
      if (unsaved) throw new Error("Your last change wasn’t saved. Make it again, then publish.");
      return post("api/publish", body);
    }).then(function (j) {
      go.disabled = false;
      if (j.error) { go.textContent = "Try again"; say(j.error, true); return; }
      published = true;
      link.value = j.url; act.hidden = true; done.hidden = false;
      say(j.protected ? "Anyone with the link and the passcode can view it." : "Anyone with the link can view it.");
    }, function (e) { go.disabled = false; go.textContent = "Try again"; say(e.message, true); });
  });
  copy.addEventListener("click", function () {
    function ok() { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy"; }, 2000); }
    if (navigator.clipboard) navigator.clipboard.writeText(link.value).then(ok, function () { link.select(); });
    else { link.select(); }
  });
  link.addEventListener("focus", function () { link.select(); });

  lockUi(); wire();
  vb = d.createElement("button"); vb.type = "button"; vb.className = "keep lb-hide"; vb.title = "Hide (H)";
  d.querySelector(".lb").appendChild(vb);
  vb.addEventListener("click", function () { var a = fdx.current(); if (a) toggle(a.closest(".ph[data-rel]").dataset.rel); });
  fdx.onshow = label;
})();
