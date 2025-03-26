#include <uuid.h>

#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

#include "utils/FeatureDB.h"
#include "RGOR.h"

int main() {
  std::unordered_map<int, int> a;

  FeatureDB db(32);

  std::vector<Eigen::VectorXf> features;
  for (int i = 0; i < 10; ++i) {
    Eigen::VectorXf vec(32);
    vec.setRandom();
    features.push_back(vec);
  }
  uuids::uuid uuid_;
  for (int i = 0; i < features.size(); ++i) {
    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 engine(seq);
    uuids::uuid_random_generator gen(&engine);
    uuid_ = gen();
    std::cout << uuid_ << std::endl;
    db.add_feature(uuid_, features[i]);
  }
  db.remove_feature(uuid_);
  auto ret = db.search_knn(features[0], 20);

  for (auto &[uuid, score] : ret) {
    std::cout << uuid << " " << score << std::endl;
  }
  db.save_index("test.index");

  return 0;
}