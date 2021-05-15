import dl_translate as dlt


# Returns True if text is spun, returns False if text is equal
def evaluate_output(input, output):
    if input == output:
        return False
    else:
        return True


# Translates "input_text" in language "src_lang" to language "tgt_lang" and back to "src_lang" using given "model"
def spin(model, src_lang, tgt_lang, input_text):
    intermediate = model.translate(input_text, source=src_lang, target=tgt_lang)  # Intermediate sentence
    return model.translate(intermediate, source=tgt_lang, target=src_lang)  # Result sentence


# Prints input, output and evaluation line by line
def print_results(model, src_lang, tgt_lang, input_text):
    output = spin(model=model, src_lang=src_lang, tgt_lang=tgt_lang, input_text=input_text)
    print("-------------------------------------------------------------------------------------------------------")
    print("Input:  " + input_text)
    print("Output: " + output)
    print("Spun:   " + str(evaluate_output(input_text, output)))



model = dlt.TranslationModel("mbart50")
src_lang = "German"
tgt_lang = "English"
input_text = "Der UN-Generalsekretär sprach auf der letzten Pressekonferenz von Angriffen auf die Persönlichkeitsrechte der Bürger Südamerikas."

print_results(model=model, src_lang=src_lang,tgt_lang=tgt_lang,input_text=input_text)

