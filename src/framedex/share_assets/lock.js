// fdx share: open a passcode-protected album in the browser. The page body
// and every image are AES-GCM ciphertext; the key comes from the passcode via
// PBKDF2. Nothing is sent anywhere: wrong passcodes fail locally.
(function () {
  var d = document, fdx = window.fdx;
  var data = JSON.parse(d.getElementById("fdx-data").textContent);
  var lock = d.querySelector(".lock"), form = d.querySelector(".gate");
  var input = form.querySelector("input"), go = form.querySelector("button");
  var msg = form.querySelector(".msg"), root = d.getElementById("g");
  var enc = new TextEncoder(), key = null, full = {};

  function bytes(b64) {
    var s = atob(b64), u = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  function open(blob, aad) {
    var u = new Uint8Array(blob);
    if (u.length < 32 || u[0] !== 70 || u[1] !== 68 || u[2] !== 88 || u[3] !== 49)
      return Promise.reject(new Error("not a sealed file"));
    return crypto.subtle.decrypt(
      { name: "AES-GCM", iv: u.slice(4, 16), additionalData: enc.encode(aad) },
      key, u.slice(16));
  }
  function fetchOpen(path) {
    return fetch(data.prefix + path)
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
      .then(function (buf) { return open(buf, path); })
      .then(function (plain) { return URL.createObjectURL(new Blob([plain], { type: "image/jpeg" })); });
  }

  if (!(window.crypto && crypto.subtle && window.TextEncoder)) {
    msg.textContent = "Open this link in an up-to-date browser over https to view the album.";
    input.disabled = true;
    return;
  }
  input.addEventListener("input", function () { go.disabled = !input.value; msg.textContent = ""; });
  // Focus on desktops only: on a phone it would throw the keyboard up
  // over the cover before anyone asked for it.
  if (window.matchMedia && matchMedia("(hover: hover)").matches) input.focus();

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (!input.value || form.classList.contains("busy")) return;
    form.classList.add("busy"); msg.textContent = "";
    crypto.subtle.importKey("raw", enc.encode(input.value), "PBKDF2", false, ["deriveKey"])
      .then(function (base) {
        return crypto.subtle.deriveKey(
          { name: "PBKDF2", salt: bytes(data.salt), iterations: data.iter, hash: "SHA-256" },
          base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
      })
      .then(function (k) { key = k; return open(bytes(data.body).buffer, "fdx:manifest"); })
      .then(show, function () {
        key = null; form.classList.remove("busy");
        msg.textContent = "That passcode didn’t work.";
        form.classList.remove("shake"); void form.offsetWidth; form.classList.add("shake");
        input.select();
      });
  });

  function show(plain) {
    root.innerHTML = new TextDecoder().decode(plain);
    root.hidden = false;
    lock.classList.add("gone");
    setTimeout(function () { lock.remove(); }, 700);
    var imgs = [].slice.call(root.querySelectorAll("img[data-thumb]"));
    function load(im) {
      var wide = im.getBoundingClientRect().width * (window.devicePixelRatio || 1);
      var path = wide > +im.dataset.tw * 1.05 ? im.dataset.photo : im.dataset.thumb;
      fetchOpen(path).then(function (url) {
        im.onload = function () { im.classList.add("in"); };
        im.src = url;
      }, function () {
        im.classList.add("in", "err");
        im.title = "This photo couldn’t be opened.";
      });
    }
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); load(e.target); } });
      }, { rootMargin: "1200px 0px" });
      imgs.forEach(function (im) { io.observe(im); });
    } else imgs.forEach(load);
    fdx.full = function (a) {
      var path = a.querySelector("img").dataset.photo;
      if (!full[path]) full[path] = fetchOpen(path);
      return full[path];
    };
    fdx.bind(root);
  }
})();
