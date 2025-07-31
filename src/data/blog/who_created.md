---

title: "Who created universe"
pubDatetime: 2025-06-31T11:30:00Z
description: "If we are created, who created the creator?"
tags: [for-loop]
------------------------------------------------------------------------
```python
def who_created(us):
    if us.is_created() == False:
        return "We just are."
    else:
        creator = us.creator()
        return who_created(creator)

# Start from our universe
who_created(universe)
```