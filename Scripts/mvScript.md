#  mvScript

## Intro

// mvScript is short for melodyVerse Script

-   mvScript is somehow like basic, our syntax is like this:
    *   define constants and marco in $yourname.h
    *   write command in .md file with markdown syntax
    *   in JiangHu, we'll first support those commands:
        -   $name **move to** [loc]
        -   $name **talk to** $name
        -   $name **say** [sth]
        -   $name **meet** $name
        -   $name **like** $name
        -   $name **love** $name
        -   $name **engage with** $name
        -   $name **hate** $name
        -   $name **fight with** $name
        -   $name **gain** [treasure]
        -   $name **learn** [skill]
        -   $name **beat** $name
        -   $name **kill** $name
        -   $name **marry** $name
        -   $name **die**

    *   And we'll support boolean/String/Int like this:
    ```swift
        var isMeteor = true
        let currentChar = "wuming"
        var hp = 120
    ```

##  Sample

// test.md

```swift
    let currentChar = "吴茗"
    let currentCity = "江城"
    if currentChar.age == 20:   // 少年出山
        $currentChar move to $currentCity 
```

