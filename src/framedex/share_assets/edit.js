// fdx share editor: the gallery itself is the workspace. Tap to hide or
// restore, drag (or Alt+arrows) to reorder within a day, click the title to
// rename, pick the cover, set a passcode, Publish. Every change is saved to
// selection.txt by the local server, which returns the re-rendered page.
(function () {
  var d = document, fdx = window.fdx;
  var cfg = JSON.parse(d.getElementById("fdx-edit").textContent);
  var g = d.getElementById("g"), bar = d.querySelector(".bar");
  var note = bar.querySelector(".note"), hid = bar.querySelector(".hid");
  var act = bar.querySelector(".act"), done = bar.querySelector(".done");
  var lockon = bar.querySelector(".lockon"), pass = bar.querySelector(".pass");
  var code = bar.querySelector(".code"), show = bar.querySelector(".show");
  var go = bar.querySelector(".go"), link = bar.querySelector(".link"), copy = bar.querySelector(".copy");
  var cover = cfg.cover, queue = Promise.resolve(), dragRel = null, published = false;

  function say(text, err) { note.textContent = text || ""; note.classList.toggle("err", !!err); }
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
    var payload = { title: title(), cover: cover, sections: model() };
    if (edit) edit(payload);
    queue = queue.then(function () { return post("api/save", payload); }).then(function (j) {
      g.innerHTML = j.html; wire(); d.title = "Editing \u00b7 " + title();
      hid.textContent = j.hidden ? " " + j.hidden + " hidden." : "";
      if (before !== null && q(anchor)) window.scrollBy(0, q(anchor).getBoundingClientRect().top - before);
      say(""); changed();
    }, function (e) { say("Couldn’t save that change: " + e.message, true); });
    return queue;
  }
  function changed() {
    if (!published) return;
    done.hidden = true; act.hidden = false; go.textContent = "Publish changes";
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
    fdx.reveal(g);
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
      var tools = d.createElement("div"); tools.className = "tools";
      if (hero) {
        tools.innerHTML = '<button type="button" class="t-uncover">Remove cover</button>';
        var tag = d.createElement("span"); tag.className = "cover-tag"; tag.textContent = "Cover";
        fig.appendChild(tag);
      } else if (!fig.classList.contains("off")) {
        tools.innerHTML = '<button type="button" class="t-cover">Make cover</button>';
      }
      fig.appendChild(tools);
      if (fig.classList.contains("off")) {
        var b = d.createElement("span"); b.className = "badge"; b.textContent = "Hidden · Tap to restore";
        fig.appendChild(b);
      }
      a.setAttribute("aria-label", (fig.classList.contains("off") ? "Restore " : "Hide ") + "photo");
      tools.addEventListener("click", function (e) {
        e.stopPropagation();
        if (e.target.classList.contains("t-cover")) { cover = rel; save(rel); }
        if (e.target.classList.contains("t-uncover")) { cover = "none"; save(null); }
      });
      a.addEventListener("click", function (e) {
        e.preventDefault();
        if (hero) return;
        fig.classList.toggle("off"); save(rel);
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
      fig.addEventListener("dragend", function () { fig.classList.remove("dragging"); clearMarks(); dragRel = null; });
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
    queue.then(function () { return post("api/publish", body); }).then(function (j) {
      go.disabled = false;
      if (j.error) { go.textContent = "Try again"; say(j.error, true); return; }
      published = true; cfg.protected = j.protected;
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
})();
