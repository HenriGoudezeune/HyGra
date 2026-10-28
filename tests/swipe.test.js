// Spec-driven tests for WheelPager (web/swipe.js). Written from the JSDoc spec, not from the implementation.
// Run from C:\Users\henri\ncm-code with:  node --test tests/

const test = require("node:test");
const assert = require("node:assert/strict");
const WheelPager = require("../web/swipe.js");

// Event builder. Defaults: a page in the middle, neighbours on both sides, no inner scroller.
function ev(over = {}) {
  return Object.assign(
    { dx: 0, dy: 0, t: 0, atTop: false, atBottom: false, canPrev: false, canNext: false, innerX: false },
    over
  );
}

// Pager with deck width 1000 px unless overridden: swipe threshold 200 px, edge 240, quiet 180, settle 320.
function pager(opts = {}) {
  return new WheelPager(Object.assign({ width: 1000 }, opts));
}

const NONE_FREE = { kind: "none", prevent: false };
const NONE_LOCKED = { kind: "none", prevent: true };

// Floating-point tolerance for the pull values (0.35 multiplier).
function near(actual, expected, msg) {
  assert.ok(
    Math.abs(actual - expected) < 1e-9,
    `${msg || "value"}: expected ${expected} (within 1e-9), got ${actual}`
  );
}

function assertPull(r, y, msg) {
  assert.equal(r.kind, "pull", `${msg || "pull"}: kind`);
  assert.equal(r.prevent, true, `${msg || "pull"}: prevent`);
  near(r.y, y, `${msg || "pull"}: y`);
}

// ---------------------------------------------------------------------------
// Horizontal (rule 3)
// ---------------------------------------------------------------------------

test("horizontal toward next: drags with x = -accX, then goes dir 1 axis x once accX reaches 0.2*width", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 0, canNext: true })), { kind: "drag", x: -100, prevent: true });
  assert.deepEqual(p.wheel(ev({ dx: 50, t: 20, canNext: true })), { kind: "drag", x: -150, prevent: true });
  // accX = 200 = 0.2 * 1000: commits the move.
  assert.deepEqual(p.wheel(ev({ dx: 50, t: 40, canNext: true })), { kind: "go", dir: 1, axis: "x", prevent: true });
});

test("horizontal toward previous (dx < 0): drags with positive x, then goes dir -1 axis x", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dx: -100, t: 0, canPrev: true })), { kind: "drag", x: 100, prevent: true });
  assert.deepEqual(p.wheel(ev({ dx: -100, t: 20, canPrev: true })), { kind: "go", dir: -1, axis: "x", prevent: true });
});

test("horizontal: no move while |accX| is still below 0.2*width", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 0, canNext: true })), { kind: "drag", x: -100, prevent: true });
  assert.deepEqual(p.wheel(ev({ dx: 99, t: 20, canNext: true })), { kind: "drag", x: -199, prevent: true });
});

test("horizontal: no neighbour in the swipe direction gives a clamped rubber band and never a go", () => {
  const p = pager();
  // canNext is false: x = -accX * 0.25
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 0, canPrev: true })), { kind: "drag", x: -25, prevent: true });
  // accX = 200 (past the threshold), still no next page: no go
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 20, canPrev: true })), { kind: "drag", x: -50, prevent: true });
  // accX = 300: -75 clamped to -60
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 40, canPrev: true })), { kind: "drag", x: -60, prevent: true });
  // and it keeps clamping, never a go
  assert.deepEqual(p.wheel(ev({ dx: 400, t: 60, canPrev: true })), { kind: "drag", x: -60, prevent: true });
});

test("horizontal: rubber band mirrors toward the previous end (positive x, clamped to +60)", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dx: -100, t: 0, canNext: true })), { kind: "drag", x: 25, prevent: true });
  assert.deepEqual(p.wheel(ev({ dx: -400, t: 20, canNext: true })), { kind: "drag", x: 60, prevent: true });
});

test("horizontal: innerX returns none/prevent false and the delta does not count toward the swipe", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dx: 150, t: 0, canNext: true, innerX: true })), NONE_FREE);
  // If the 150 had been accumulated, this 100 would already be a go. It must be a plain drag.
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 20, canNext: true })), { kind: "drag", x: -100, prevent: true });
});

test("horizontal: a new gesture (gap > 180 ms) resets the horizontal accumulator", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dx: 150, t: 0, canNext: true })), { kind: "drag", x: -150, prevent: true });
  // gap 300 ms: new gesture, accX starts from 0 again, so 150 is not a go
  assert.deepEqual(p.wheel(ev({ dx: 150, t: 300, canNext: true })), { kind: "drag", x: -150, prevent: true });
});

test("horizontal: |dx| == |dy| is not horizontal, so the vertical branch handles it", () => {
  const p = pager();
  // Vertical branch: accY = 30, pull = 30 * 0.35
  assertPull(p.wheel(ev({ dx: 30, dy: -30, t: 0, atTop: true, canPrev: true })), 10.5, "tie goes vertical");
});

// ---------------------------------------------------------------------------
// Gesture lock and gesture boundaries (rules 1 and 2)
// ---------------------------------------------------------------------------

test("after a go, further events in the same gesture return none/prevent true", () => {
  const p = pager();
  p.wheel(ev({ dx: 100, t: 0, canNext: true }));
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 20, canNext: true })), { kind: "go", dir: 1, axis: "x", prevent: true });
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 60, canNext: true })), NONE_LOCKED);
  assert.deepEqual(p.wheel(ev({ dx: -300, t: 100, canPrev: true })), NONE_LOCKED);
  assert.deepEqual(p.wheel(ev({ dy: -100, t: 120, atTop: true, canPrev: true })), NONE_LOCKED);
});

test("gap under 180 ms keeps the lock; gap over 180 ms starts a new gesture that works again", () => {
  const p = pager();
  p.wheel(ev({ dx: 100, t: 0, canNext: true }));
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 20, canNext: true })), { kind: "go", dir: 1, axis: "x", prevent: true });
  // gap 130 ms from the last event: still the same gesture, still locked
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 150, canNext: true })), NONE_LOCKED);
  // gap 200 ms: new gesture, unlocked
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 350, canNext: true })), { kind: "drag", x: -100, prevent: true });
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 370, canNext: true })), { kind: "go", dir: 1, axis: "x", prevent: true });
});

test("a go in the vertical direction also locks the rest of the gesture", () => {
  const p = pager();
  p.wheel(ev({ dy: -80, t: 0, atTop: true, canPrev: true }));
  p.wheel(ev({ dy: -80, t: 20, atTop: true, canPrev: true }));
  assert.deepEqual(
    p.wheel(ev({ dy: -80, t: 40, atTop: true, canPrev: true })),
    { kind: "go", dir: -1, axis: "y", prevent: true }
  );
  assert.deepEqual(p.wheel(ev({ dy: -80, t: 60, atTop: true, canPrev: true })), NONE_LOCKED);
});

// ---------------------------------------------------------------------------
// Vertical, pulling past the top (rule 4, dy < 0)
// ---------------------------------------------------------------------------

test("vertical: scrolling up from the top pulls (positive y, 0.35 * accY), then goes dir -1 axis y at accY 240", () => {
  const p = pager();
  assertPull(p.wheel(ev({ dy: -80, t: 0, atTop: true, canPrev: true })), 28, "first notch-ish");
  assertPull(p.wheel(ev({ dy: -80, t: 20, atTop: true, canPrev: true })), 56, "second");
  // accY = 240 = edge default
  assert.deepEqual(
    p.wheel(ev({ dy: -80, t: 40, atTop: true, canPrev: true })),
    { kind: "go", dir: -1, axis: "y", prevent: true }
  );
});

test("vertical: the pull is capped at 90 px", () => {
  const p = pager({ edge: 1000 }); // raise the edge so the pull is not turned into a go first
  assertPull(p.wheel(ev({ dy: -300, t: 0, atTop: true, canPrev: true })), 90, "capped");
});

test("vertical: edge option moves the go threshold", () => {
  const p = pager({ edge: 100 });
  assert.deepEqual(
    p.wheel(ev({ dy: -100, t: 0, atTop: true, canPrev: true })),
    { kind: "go", dir: -1, axis: "y", prevent: true }
  );
});

test("vertical: a gesture that did not start at the top never pulls or goes, even when later events are at the top", () => {
  const p = pager();
  // momentum reaching the top: the first event of the gesture is not at the top
  assert.deepEqual(p.wheel(ev({ dy: -100, t: 0, atTop: false, canPrev: true })), NONE_FREE);
  for (let i = 1; i <= 5; i++) {
    assert.deepEqual(
      p.wheel(ev({ dy: -100, t: 20 * i, atTop: true, canPrev: true })),
      NONE_FREE,
      `event ${i} at top, same gesture`
    );
  }
});

test("vertical: leaving the top mid-gesture clears the pull; returning to the top starts from zero", () => {
  const p = pager();
  assertPull(p.wheel(ev({ dy: -100, t: 0, atTop: true, canPrev: true })), 35, "pull");
  assert.deepEqual(p.wheel(ev({ dy: -100, t: 20, atTop: false, canPrev: true })), NONE_FREE);
  // accY was cleared, so this is 100 again (pull 35), not 200 (pull 70)
  assertPull(p.wheel(ev({ dy: -100, t: 40, atTop: true, canPrev: true })), 35, "after clear");
});

test("vertical: scrolling up at the top with no previous page falls back to native scrolling", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dy: -100, t: 0, atTop: true, canNext: true })), NONE_FREE);
});

// ---------------------------------------------------------------------------
// Vertical, pulling past the bottom (rule 4, dy > 0)
// ---------------------------------------------------------------------------

test("vertical: scrolling down at the bottom pulls (negative y), then goes dir 1 axis y at accY 240", () => {
  const p = pager();
  assertPull(p.wheel(ev({ dy: 80, t: 0, atBottom: true, canNext: true })), -28, "pull down");
  assertPull(p.wheel(ev({ dy: 80, t: 20, atBottom: true, canNext: true })), -56, "pull down 2");
  assert.deepEqual(
    p.wheel(ev({ dy: 80, t: 40, atBottom: true, canNext: true })),
    { kind: "go", dir: 1, axis: "y", prevent: true }
  );
});

test("vertical: scrolling down at the bottom with no next page falls back to native scrolling", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dy: 100, t: 0, atBottom: true, canPrev: true })), NONE_FREE);
});

test("vertical: reversing direction within a gesture resets the accumulator", () => {
  const p = pager();
  // page fits the viewport: both edges, both neighbours
  const both = { atTop: true, atBottom: true, canPrev: true, canNext: true };
  assertPull(p.wheel(ev(Object.assign({ dy: 100, t: 0 }, both))), -35, "down first");
  // up: accY was collected in the other direction, so it restarts at 100 (pull +35), not 200
  assertPull(p.wheel(ev(Object.assign({ dy: -100, t: 20 }, both))), 35, "up after down");
});

// ---------------------------------------------------------------------------
// Vertical, normal scrolling and zero deltas (rules 4 and 5)
// ---------------------------------------------------------------------------

test("normal scrolling in the middle of a page is left to the browser", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dy: -40, t: 0, canPrev: true, canNext: true })), NONE_FREE);
  assert.deepEqual(p.wheel(ev({ dy: 40, t: 20, canPrev: true, canNext: true })), NONE_FREE);
});

test("dx == dy == 0 returns none/prevent false", () => {
  const p = pager();
  const e = { atTop: true, atBottom: true, canPrev: true, canNext: true };
  assert.deepEqual(p.wheel(ev(Object.assign({ dx: 0, dy: 0, t: 0 }, e))), NONE_FREE);
  // still nothing in progress afterwards
  assert.deepEqual(p.wheel(ev(Object.assign({ dx: 0, dy: 0, t: 20 }, e))), NONE_FREE);
});

// ---------------------------------------------------------------------------
// Mouse wheel notches and settle (rule 1, accY reset rule)
// ---------------------------------------------------------------------------

test("slow mouse-wheel notches (250 ms apart, within settle) add up to a go on the 3rd notch", () => {
  const p = pager();
  assertPull(p.wheel(ev({ dy: -100, t: 0, atTop: true, canPrev: true })), 35, "notch 1");
  assertPull(p.wheel(ev({ dy: -100, t: 250, atTop: true, canPrev: true })), 70, "notch 2");
  assert.deepEqual(
    p.wheel(ev({ dy: -100, t: 500, atTop: true, canPrev: true })),
    { kind: "go", dir: -1, axis: "y", prevent: true }
  );
});

test("notches more than 320 ms apart reset accY: no go after 3 notches", () => {
  const p = pager();
  assertPull(p.wheel(ev({ dy: -100, t: 0, atTop: true, canPrev: true })), 35, "notch 1");
  assertPull(p.wheel(ev({ dy: -100, t: 350, atTop: true, canPrev: true })), 35, "notch 2 (reset)");
  assertPull(p.wheel(ev({ dy: -100, t: 700, atTop: true, canPrev: true })), 35, "notch 3 (reset)");
});

// ---------------------------------------------------------------------------
// idle(t) (rule set for the settle timer)
// ---------------------------------------------------------------------------

test("idle snaps a partial horizontal drag back once 320 ms have passed since the last event", () => {
  const p = pager();
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 1000, canNext: true })), { kind: "drag", x: -100, prevent: true });
  // too early: nothing happens, and the drag is still pending
  assert.deepEqual(p.idle(1000 + 319), { kind: "none" });
  assert.deepEqual(p.idle(1000 + 320), { kind: "snap" });
  // reset: nothing in progress any more
  assert.deepEqual(p.idle(2000), { kind: "none" });
});

test("idle measures the gap from the last wheel event, not from the start of the gesture", () => {
  const p = pager();
  p.wheel(ev({ dx: 100, t: 0, canNext: true }));
  p.wheel(ev({ dx: 50, t: 100, canNext: true })); // lastT = 100
  assert.deepEqual(p.idle(400), { kind: "none" }); // 300 ms since last event
  assert.deepEqual(p.idle(420), { kind: "snap" }); // 320 ms since last event
});

test("idle snaps a partial vertical pull back too", () => {
  const p = pager();
  assertPull(p.wheel(ev({ dy: -100, t: 0, atTop: true, canPrev: true })), 35, "pull");
  assert.deepEqual(p.idle(320), { kind: "snap" });
  assert.deepEqual(p.idle(400), { kind: "none" });
});

test("idle returns none when nothing is in progress", () => {
  const p = pager();
  assert.deepEqual(p.idle(5000), { kind: "none" });
});

test("idle returns none after a completed go (locked, nothing accumulated)", () => {
  const p = pager();
  p.wheel(ev({ dx: 100, t: 0, canNext: true }));
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 20, canNext: true })), { kind: "go", dir: 1, axis: "x", prevent: true });
  assert.deepEqual(p.idle(10000), { kind: "none" });
});

// ---------------------------------------------------------------------------
// Constructor options and the width setter
// ---------------------------------------------------------------------------

test("width can be changed after construction: the swipe threshold follows .width", () => {
  const p = new WheelPager({ width: 1000 });
  p.width = 500; // threshold is now 0.2 * 500 = 100
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 0, canNext: true })), { kind: "go", dir: 1, axis: "x", prevent: true });
});

test("swipe option replaces the 0.2 fraction", () => {
  const p = new WheelPager({ width: 1000, swipe: 0.5 }); // threshold 500
  assert.deepEqual(p.wheel(ev({ dx: 400, t: 0, canNext: true })), { kind: "drag", x: -400, prevent: true });
  assert.deepEqual(p.wheel(ev({ dx: 100, t: 20, canNext: true })), { kind: "go", dir: 1, axis: "x", prevent: true });
});

test("quiet option replaces the 180 ms gesture gap", () => {
  const p = new WheelPager({ width: 1000, quiet: 100 });
  assert.deepEqual(p.wheel(ev({ dx: 150, t: 0, canNext: true })), { kind: "drag", x: -150, prevent: true });
  // gap 150 > quiet 100: new gesture, accX restarts; with the default 180 this would be a go
  assert.deepEqual(p.wheel(ev({ dx: 150, t: 150, canNext: true })), { kind: "drag", x: -150, prevent: true });
});

test("settle option replaces the 320 ms accY reset gap", () => {
  const p = new WheelPager({ width: 1000, quiet: 50, settle: 100 });
  assertPull(p.wheel(ev({ dy: -100, t: 0, atTop: true, canPrev: true })), 35, "notch 1");
  // gap 150 > settle 100: accY resets (with the default 320 it would accumulate to a pull of 70)
  assertPull(p.wheel(ev({ dy: -100, t: 150, atTop: true, canPrev: true })), 35, "notch 2");
  assertPull(p.wheel(ev({ dy: -100, t: 300, atTop: true, canPrev: true })), 35, "notch 3");
});
