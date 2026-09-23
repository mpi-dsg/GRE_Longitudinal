#pragma once
// A real bulk loader for BTreeOLC, with a settable leaf fill factor.
//
// Why this exists: BTreeOLC ships no bulk load. GRE's wrapper `bulk_load()` is a loop of
// ordinary inserts, so it never creates the uniformly-filled page cohort that "waves of
// misery" is actually about. To run Glombiewski et al.'s randomized-fill remedy on a
// B+tree -- their own structure, with optimistic lock coupling added -- we first have to
// build that cohort.
//
// Leaves are filled to `fill` (uniform) or to a draw from [fill-spread, fill+spread]
// (randomized). Inner nodes are always filled to a constant, so the leaf fill is the only
// thing that differs between arms.
//
// Separator semantics, from BTreeOLC_child_layout.h: descent is
// children[lowerBound(k)], and lowerBound returns the first index with keys[i] >= k, so
// keys[i] is the MAX key of subtree children[i], with count separators and count+1 children.
#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace btree_bulk {

inline bool &randomized() { static bool r = false; return r; }
inline double &fill()     { static double f = 0.70; return f; }
inline double &spread()   { static double s = 0.0;  return s; }
inline unsigned &seed()   { static unsigned s = 12345; return s; }

template <class Key, class Val>
void load(btreeolc::BTree<Key, Val> &t, const std::pair<Key, Val> *kv, size_t n,
          unsigned seed) {
  using Leaf = btreeolc::BTreeLeaf<Key, Val>;
  using Inner = btreeolc::BTreeInner<Key>;
  if (n == 0) return;

  const size_t Lmax = Leaf::maxEntries;
  const size_t Imax = Inner::maxEntries;
  std::mt19937_64 rng(seed);

  std::vector<btreeolc::NodeBase *> level;
  std::vector<Key> level_max;

  Leaf *prev = nullptr;
  for (size_t i = 0; i < n;) {
    double f = fill();
    if (randomized() && spread() > 0.0)
      f = std::uniform_real_distribution<double>(fill() - spread(),
                                                 fill() + spread())(rng);
    f = std::min(1.0, std::max(0.05, f));
    size_t take = std::min(n - i, std::max<size_t>(1, (size_t)(Lmax * f)));
    Leaf *l = new Leaf();
    for (size_t j = 0; j < take; j++) l->data[j] = kv[i + j];
    l->count = (uint16_t)take;
    if (prev) prev->next_leaf = l;
    prev = l;
    level.push_back(l);
    level_max.push_back(kv[i + take - 1].first);
    i += take;
  }

  while (level.size() > 1) {
    std::vector<btreeolc::NodeBase *> up;
    std::vector<Key> up_max;
    size_t group = std::max<size_t>(2, std::min(Imax, (size_t)(Imax * 0.70)));
    for (size_t s = 0; s < level.size();) {
      size_t e = std::min(level.size(), s + group);
      // Never leave a trailing group of one: an inner node with count==0 makes
      // lowerBound read keys[0] uninitialized and can return an out-of-range child.
      if (level.size() - e == 1 && e - s >= 3) e = level.size();
      size_t cnt = e - s;
      Inner *in = new Inner();
      for (size_t j = 0; j < cnt; j++) in->children[j] = level[s + j];
      for (size_t j = 0; j + 1 < cnt; j++) in->keys[j] = level_max[s + j];
      in->count = (uint16_t)(cnt - 1);
      up.push_back(in);
      up_max.push_back(level_max[e - 1]);
      s = e;
    }
    level.swap(up);
    level_max.swap(up_max);
  }
  t.root.store(level[0]);
}

}  // namespace btree_bulk
