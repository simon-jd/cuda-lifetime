// No CUDA: an uninstantiated template's dependent calls have no runtime body.
template<class T> struct HostTraits {
  static constexpr int max_digits = T::digits();
};
template<class T> constexpr int dependent_value = T::value();
int host_template_only() { return 42; }
