def label_conversion(inline_labels: list[str], inter_labels: list[str], confidences: list[float] = None) -> list[str]:
    """
    Func:
        Given the fine grain label of new models, convert them to old labels
    Args:   
        inline_labels: list[str], have label: keep, replace, delete
        inter_labels: list[str], have label: null, insert, block-split
    Return:
        old_labels: list[str]
    """
    assert len(inline_labels) + 1 == len(inter_labels)
    # rule 1: block-split can be ignored
    inter_labels = ["<null>" if x == "<block-split>" else x for x in inter_labels]
    
    # rule 2: delete is a part of replace
    inline_labels = ["<replace>" if x == "<delete>" else x for x in inline_labels]
    
    # rule 3: old labels can't handle insert at the beginning of code window
    inter_labels = inter_labels[1:]
    
    old_labels = []
    old_confidences = []
    # rule 4: now inter_label  should only have null & insert
    #             inline_label should only have keep & replace
    if confidences is None:
        for idx, (inter_label, inline_label) in enumerate(zip(inter_labels, inline_labels)):
            if inter_label == "<null>":
                old_labels.append(inline_label)
            else: # inter_label == "insert"
                if inline_label == "<keep>":
                    old_labels.append("<insert>")
                else:
                    old_labels.append("<replace>")
        assert "<null>" not in old_labels
        assert "<block-split>" not in old_labels
        assert "<delete>" not in old_labels
        assert len(old_labels) == len(inline_labels)
        return old_labels
    else:
        inter_label_confidences = [confidences[i] for i in range(0, len(confidences), 2)]
        inline_label_confidences = [confidences[i] for i in range(1, len(confidences), 2)]
        for idx, (inter_label, inline_label, inter_conf, inline_conf) in enumerate(zip(inter_labels, inline_labels, inter_label_confidences, inline_label_confidences)):
            if inter_label == "<null>":
                old_labels.append(inline_label)
                old_confidences.append(inline_conf)
            else: # inter_label == "insert"
                if inline_label == "<keep>":
                    old_labels.append("<insert>")
                    old_confidences.append(inter_conf)
                else:
                    old_labels.append("<replace>")
                    old_confidences.append(inline_conf)
        assert "<null>" not in old_labels
        assert "<block-split>" not in old_labels
        assert "<delete>" not in old_labels
        assert len(old_labels) == len(inline_labels)
        assert len(old_labels) == len(old_confidences)
        return old_labels, old_confidences
                
    
                