__all__ = ("Character",)


class Character:
    def __init__(self, full_name: str, gender: str, aliases: str):
        name, jp_name = full_name.strip(')').rsplit(' (', 1)
        self.name = name
        self.japanese_name = jp_name
        if "・" in jp_name:
            if " " in name:
                self.fp_name = name.split()[0]
            else:
                self.fp_name = name
            self.japanese_fp_name = jp_name.split('・')[0]
        else:
            if " " in name:
                self.fp_name = name.split()[1]
            else:
                self.fp_name = name
            if " " in jp_name:
                self.japanese_fp_name = jp_name.split()[1]
            else:
                self.japanese_fp_name = jp_name
        self.gender = gender
        self.aliases = aliases