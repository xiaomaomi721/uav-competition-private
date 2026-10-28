# A05 residual-D1 matched pair templates

用于：

```text
A05_D1HARD_CONTROL_S20260922_L2000
A05_D1HARD_TREAT_S20260922_L2000
```

关键规则：

- parent必须是SHA绑定的A05，不得重建；
- teacher caches必须是theta261与original theta278-export；
- control完整执行target分支，只把target hard loss权重设为0；
- formal固定2000 attempted loops，AMP skip不补；
- loop1000不可选择/提交，loop2000是唯一endpoint；
- 无效果型pilot/gate；工程smoke通过后两臂完整执行；
- 只允许物理GPU2；
- 两包都是单checkpoint；
- pair不可拆分提交，用户手工control→treatment；
- 无执行报告或pair verification时不得采信。
