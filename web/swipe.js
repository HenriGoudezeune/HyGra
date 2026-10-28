// SWIPE: turns raw wheel events (touchpad two-finger swipes, mouse wheel) into page moves for the chat deck.
// Pure logic: no DOM, no timers; the page passes the time and the scroll state in, and draws what comes back.
// HyGra uses it for horizontal swipes only: vertical scrolling between questions is a continuous strip in index.html (vertWheel).
//
// Axes: dx > 0 = toward the NEXT page (right), dx < 0 = toward the PREVIOUS page (left);
//       dy < 0 = scrolling up, dy > 0 = scrolling down (the usual WheelEvent signs).
//
// A "gesture" is a run of wheel events with gaps shorter than `quiet` ms between them.
class WheelPager {
  /**
   * @param {object} o
   * @param {number} o.width          deck width in px (updated later via .width)
   * @param {number} [o.swipe=0.2]    horizontal travel, as a fraction of width, that commits a page move
   * @param {number} [o.edge=240]     extra vertical px past a page edge that commits a move to the previous/next page
   * @param {number} [o.quiet=180]    ms without events that ends a gesture
   * @param {number} [o.settle=320]   ms without events after which an unfinished drag/pull snaps back (see idle)
   */
  constructor(o) {
    this.width = o.width;
    this.swipe = o.swipe ?? 0.2;
    this.edge = o.edge ?? 240;
    this.quiet = o.quiet ?? 180;
    this.settle = o.settle ?? 320;
    this.accX = 0;
    this.accY = 0;
    this.accDir = 0;
    this.lastT = null;
    this.locked = false;
    this.edgeTop = false;
    this.edgeBottom = false;
  }

  /**
   * One wheel event. Returns what the page must do:
   *   { kind: "none", prevent: false }             let the browser scroll natively
   *   { kind: "none", prevent: true }              swallow the event (e.g. after a move, until the gesture ends)
   *   { kind: "drag", x, prevent: true }           horizontal live offset in px for the deck: x = -accX (content follows the
   *                                                fingers); at an end without a neighbour it is a rubber band:
   *                                                x = -accX * 0.25, clamped to [-60, 60]
   *   { kind: "pull", y, prevent: true }           vertical rubber band for the page: y = +min(accY * 0.35, 90) when pulling
   *                                                past the top (toward previous), y = -min(accY * 0.35, 90) past the bottom
   *   { kind: "go", dir: -1 | 1, axis: "x" | "y", prevent: true }   move one page (dir -1 = previous, 1 = next)
   *
   * @param {object} e
   * @param {number} e.dx, e.dy       deltas in px
   * @param {number} e.t              timestamp in ms
   * @param {boolean} e.atTop         the page is scrolled to its top (scrollTop <= 1)
   * @param {boolean} e.atBottom      the page is scrolled to its bottom
   * @param {boolean} e.canPrev       a previous page exists
   * @param {boolean} e.canNext       a next page exists
   * @param {boolean} e.innerX        the pointer is over an element (code block, table) that can still scroll in the dx direction
   *
   * Rules:
   *  1. New gesture (t - lastT > quiet, or first event): unlock; remember atTop/atBottom as edgeTop/edgeBottom for this gesture;
   *     reset accX; reset accY only if t - lastT > settle (so slow mouse-wheel notches at an edge still add up).
   *     Always set lastT = t at the end.
   *  2. Locked (a "go" already fired in this gesture): return { kind: "none", prevent: true }.
   *  3. Horizontal when |dx| > |dy|:
   *     - innerX → { kind: "none", prevent: false }
   *     - accX += dx; if |accX| >= swipe * width and the neighbour in sign(accX) exists → lock, accX = 0, return go (axis "x")
   *     - neighbour exists → drag with x = -accX; no neighbour → rubber-band drag (see above)
   *  4. Vertical otherwise (dy != 0):
   *     - dy < 0 and edgeTop and atTop and canPrev: if accY was collected in the other direction (accDir 1) reset it;
   *       accY += -dy, accDir = -1; accY >= edge → lock, accY = 0, go dir -1 axis "y"; else pull with positive y
   *     - dy > 0 and edgeBottom and atBottom and canNext: mirror image (accDir 1, go dir 1, pull with negative y)
   *     - anything else: accY = 0 and { kind: "none", prevent: false }   (normal scrolling inside the page)
   *  5. dx == dy == 0 → { kind: "none", prevent: false }
   */
  wheel(e) {
    // Rule 1: new gesture (first event, or a pause longer than quiet)
    if (this.lastT === null || e.t - this.lastT > this.quiet) {
      this.locked = false;
      this.edgeTop = e.atTop;
      this.edgeBottom = e.atBottom;
      if (this.lastT === null || e.t - this.lastT > this.settle) this.accY = 0;
      this.accX = 0;
    }
    this.lastT = e.t;

    // Rule 2: a go already fired in this gesture
    if (this.locked) return { kind: "none", prevent: true };

    // Rule 3: horizontal
    if (Math.abs(e.dx) > Math.abs(e.dy)) {
      if (e.innerX) return { kind: "none", prevent: false };
      this.accX += e.dx;
      const dir = this.accX > 0 ? 1 : -1;
      const neighbour = dir > 0 ? e.canNext : e.canPrev;
      if (Math.abs(this.accX) >= this.swipe * this.width && neighbour) {
        this.locked = true;
        this.accX = 0;
        return { kind: "go", dir, axis: "x", prevent: true };
      }
      if (neighbour) return { kind: "drag", x: -this.accX, prevent: true };
      return { kind: "drag", x: Math.max(-60, Math.min(60, -this.accX * 0.25)), prevent: true };
    }

    // Rule 5: nothing moved
    if (e.dy === 0) return { kind: "none", prevent: false };

    // Rule 4: vertical, past the top
    if (e.dy < 0 && this.edgeTop && e.atTop && e.canPrev) {
      if (this.accDir === 1) this.accY = 0;
      this.accY += -e.dy;
      this.accDir = -1;
      if (this.accY >= this.edge) {
        this.locked = true;
        this.accY = 0;
        return { kind: "go", dir: -1, axis: "y", prevent: true };
      }
      return { kind: "pull", y: Math.min(this.accY * 0.35, 90), prevent: true };
    }

    // Rule 4: vertical, past the bottom
    if (e.dy > 0 && this.edgeBottom && e.atBottom && e.canNext) {
      if (this.accDir === -1) this.accY = 0;
      this.accY += e.dy;
      this.accDir = 1;
      if (this.accY >= this.edge) {
        this.locked = true;
        this.accY = 0;
        return { kind: "go", dir: 1, axis: "y", prevent: true };
      }
      return { kind: "pull", y: -Math.min(this.accY * 0.35, 90), prevent: true };
    }

    // Rule 4: normal scrolling inside the page
    this.accY = 0;
    return { kind: "none", prevent: false };
  }

  /**
   * Called by the page on a timer. If t - lastT >= settle and a drag or pull is in progress (accX or accY not 0) and not locked:
   * reset accX and accY and return { kind: "snap" } so the page animates back. Otherwise { kind: "none" }.
   */
  idle(t) {
    if (this.lastT === null) return { kind: "none" };
    if (t - this.lastT >= this.settle && (this.accX !== 0 || this.accY !== 0) && !this.locked) {
      this.accX = 0;
      this.accY = 0;
      return { kind: "snap" };
    }
    return { kind: "none" };
  }
}

if (typeof module !== "undefined") module.exports = WheelPager;
