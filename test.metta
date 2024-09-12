


(= (gatherJunctors $exp) (
    case $exp (
        ($op $a $b) (
            unify $a ($op $x $y) ($op (cons-atom $a (cons-atom $b ())))
        )
    )
))

(AND x y (AND z w))